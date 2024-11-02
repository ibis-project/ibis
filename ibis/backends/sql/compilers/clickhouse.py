from __future__ import annotations

import calendar
import math
from string import whitespace
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.sql.compilers.base import NULL, STAR, AggGen, SQLGlotCompiler
from ibis.backends.sql.datatypes import ClickHouseType
from ibis.backends.sql.dialects import ClickHouse

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class ClickhouseAggGen(AggGen):
    def aggregate(self, compiler, name, *args, where=None, order_by=()):
        if order_by:
            raise com.UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                "not supported for this backend"
            )
        # Clickhouse aggregate functions all have filtering variants with a
        # `If` suffix (e.g. `SumIf` instead of `Sum`).
        if where is not None:
            name += "If"
            args += (where,)
        return compiler.f[name](*args)


class ClickHouseCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = ClickHouse
    type_mapper = ClickHouseType

    agg = ClickhouseAggGen()

    supports_qualify = True

    UNSUPPORTED_OPS = (
        ops.RowID,
        ops.CumeDist,
        ops.PercentRank,
        ops.Time,
        ops.TimeDelta,
        ops.StringToTimestamp,
        ops.StringToDate,
        ops.StringToTime,
        ops.Levenshtein,
    )

    SIMPLE_OPS = {
        ops.All: "min",
        ops.Any: "max",
        ops.ApproxCountDistinct: "uniqHLL12",
        ops.ApproxMedian: "median",
        ops.Arbitrary: "any",
        ops.ArrayContains: "has",
        ops.ArrayFlatten: "arrayFlatten",
        ops.ArrayIntersect: "arrayIntersect",
        ops.ArrayPosition: "indexOf",
        ops.BitwiseAnd: "bitAnd",
        ops.BitwiseNot: "bitNot",
        ops.BitwiseOr: "bitOr",
        ops.BitwiseXor: "bitXor",
        ops.Capitalize: "initcap",
        ops.CountDistinct: "uniq",
        ops.Date: "toDate",
        ops.E: "e",
        ops.EndsWith: "endsWith",
        ops.ExtractAuthority: "netloc",
        ops.ExtractDay: "toDayOfMonth",
        ops.ExtractDayOfYear: "toDayOfYear",
        ops.ExtractEpochSeconds: "toRelativeSecondNum",
        ops.ExtractFragment: "fragment",
        ops.ExtractHost: "domain",
        ops.ExtractHour: "toHour",
        ops.ExtractMinute: "toMinute",
        ops.ExtractMonth: "toMonth",
        ops.ExtractPath: "path",
        ops.ExtractProtocol: "protocol",
        ops.ExtractQuarter: "toQuarter",
        ops.ExtractSecond: "toSecond",
        ops.ExtractWeekOfYear: "toISOWeek",
        ops.ExtractYear: "toYear",
        ops.ExtractIsoYear: "toISOYear",
        ops.IntegerRange: "range",
        ops.IsInf: "isInfinite",
        ops.IsNan: "isNaN",
        ops.IsNull: "isNull",
        ops.Ln: "log",
        ops.Log10: "log10",
        ops.MapKeys: "mapKeys",
        ops.MapLength: "length",
        ops.MapMerge: "mapUpdate",
        ops.MapValues: "mapValues",
        ops.Median: "quantileExactExclusive",
        ops.NotNull: "isNotNull",
        ops.NullIf: "nullIf",
        ops.RegexReplace: "replaceRegexpAll",
        ops.RowNumber: "row_number",
        ops.StartsWith: "startsWith",
        ops.StrRight: "right",
        ops.Strftime: "formatDateTime",
        ops.StringLength: "length",
        ops.StringReplace: "replaceAll",
        ops.TimestampNow: "now",
        ops.TypeOf: "toTypeName",
        ops.Unnest: "arrayJoin",
        ops.RandomUUID: "generateUUIDv4",
        ops.RandomScalar: "randCanonical",
    }

    @staticmethod
    def _minimize_spec(start, end, spec):
        if (
            start is None
            and isinstance(getattr(end, "value", None), ops.Literal)
            and end.value.value == 0
            and end.following
        ):
            return None
        return spec

    def visit_Cast(self, op, *, arg, to):
        _interval_cast_suffixes = {
            "s": "Second",
            "m": "Minute",
            "h": "Hour",
            "D": "Day",
            "W": "Week",
            "M": "Month",
            "Q": "Quarter",
            "Y": "Year",
        }

        if to.is_interval():
            suffix = _interval_cast_suffixes[to.unit.short]
            return self.f[f"toInterval{suffix}"](arg)

        result = self.cast(arg, to)
        if (timezone := getattr(to, "timezone", None)) is not None:
            return self.f.toTimeZone(result, timezone)
        return result

    def visit_TryCast(self, op, *, arg, to):
        return self.f.accurateCastOrNull(arg, self.type_mapper.to_string(to))

    def visit_ArrayIndex(self, op, *, arg, index):
        return arg[self.if_(index >= 0, index + 1, index)]

    def visit_ArrayRepeat(self, op, *, arg, times):
        param = sg.to_identifier("_")
        func = sge.Lambda(this=arg, expressions=[param])
        return self.f.arrayFlatten(self.f.arrayMap(func, self.f.range(times)))

    def visit_ArraySlice(self, op, *, arg, start, stop):
        start = self._add_parens(start)
        start_correct = self.if_(start < 0, start, start + 1)

        if stop is not None:
            stop = self._add_parens(stop)

            length = self.if_(
                stop < 0,
                stop,
                self.if_(
                    start < 0,
                    self.f.greatest(0, stop - (self.f.length(arg) + start)),
                    self.f.greatest(0, stop - start),
                ),
            )
            return self.f.arraySlice(arg, start_correct, length)
        else:
            return self.f.arraySlice(arg, start_correct)

    def visit_CountStar(self, op, *, where, arg):
        if where is not None:
            return self.f.countIf(where)
        return sge.Count(this=STAR)

    def _visit_quantile(self, func, arg, quantile, where):
        return sge.ParameterizedAgg(
            this=f"{func}If" if where is not None else func,
            expressions=util.promote_list(quantile),
            params=[arg, where] if where is not None else [arg],
        )

    def visit_Quantile(self, op, *, arg, quantile, where):
        return self._visit_quantile("quantile", arg, quantile, where)

    def visit_MultiQuantile(self, op, *, arg, quantile, where):
        return self._visit_quantile("quantiles", arg, quantile, where)

    def visit_ApproxQuantile(self, op, *, arg, quantile, where):
        return self._visit_quantile("quantileTDigest", arg, quantile, where)

    def visit_ApproxMultiQuantile(self, op, *, arg, quantile, where):
        return self._visit_quantile("quantilesTDigest", arg, quantile, where)

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "pop":
            raise ValueError(
                "ClickHouse only implements `sample` correlation coefficient"
            )
        return self.agg.corr(left, right, where=where)

    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise com.UnsupportedOperationError(
                "String find doesn't support end argument"
            )

        if start is not None:
            return self.f.position(arg, substr, start)

        return self.f.position(arg, substr)

    def visit_RegexSearch(self, op, *, arg, pattern):
        return sge.RegexpLike(this=arg, expression=pattern)

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        arg = self.cast(arg, dt.String(nullable=False))

        pattern = self.f.concat("(", pattern, ")")

        if index is None:
            index = 0

        index += 1

        then = self.f.extractGroups(arg, pattern)[index]

        return self.if_(self.f.notEmpty(then), then, NULL)

    def visit_FindInSet(self, op, *, needle, values):
        return self.f.indexOf(self.f.array(*values), needle)

    def visit_Sign(self, op, *, arg):
        """Workaround for missing sign function in older versions of clickhouse."""
        return self.f.intDivOrZero(arg, self.f.abs(arg))

    def visit_Hash(self, op, *, arg):
        return self.f.reinterpretAsInt64(self.f.sipHash64(arg))

    def visit_HashBytes(self, op, *, arg, how):
        supported_algorithms = {
            "md5": "MD5",
            "MD5": "MD5",
            "halfMD5": "halfMD5",
            "SHA1": "SHA1",
            "sha1": "SHA1",
            "SHA224": "SHA224",
            "sha224": "SHA224",
            "SHA256": "SHA256",
            "sha256": "SHA256",
            "intHash32": "intHash32",
            "intHash64": "intHash64",
            "cityHash64": "cityHash64",
            "sipHash64": "sipHash64",
            "sipHash128": "sipHash128",
        }
        if (funcname := supported_algorithms.get(how)) is None:
            raise com.UnsupportedOperationError(f"Unsupported hash algorithm {how}")

        return self.f[funcname](arg)

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        dtype = op.dtype
        if dtype.unit.short in ("ms", "us", "ns"):
            raise com.UnsupportedOperationError(
                "Clickhouse doesn't support subsecond interval resolutions"
            )
        return super().visit_IntervalFromInteger(op, arg=arg, unit=unit)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_inet():
            v = str(value)
            return self.f.toIPv6(v) if ":" in v else self.f.toIPv4(v)
        elif dtype.is_decimal():
            precision = dtype.precision
            if precision is None or not 1 <= precision <= 76:
                raise NotImplementedError(
                    f"Unsupported precision. Supported values: [1 : 76]. Current value: {precision!r}"
                )

            if 1 <= precision <= 9:
                type_name = self.f.toDecimal32
            elif 10 <= precision <= 18:
                type_name = self.f.toDecimal64
            elif 19 <= precision <= 38:
                type_name = self.f.toDecimal128
            else:
                type_name = self.f.toDecimal256
            return type_name(value, dtype.scale)
        elif dtype.is_numeric():
            if not math.isfinite(value):
                return sge.Literal.number(str(value))
            return sge.convert(value)
        elif dtype.is_interval():
            if dtype.unit.short in ("ms", "us", "ns"):
                raise com.UnsupportedOperationError(
                    "Clickhouse doesn't support subsecond interval resolutions"
                )

            return sge.Interval(
                this=sge.convert(str(value)), unit=dtype.resolution.upper()
            )
        elif dtype.is_timestamp():
            funcname = "parseDateTime"

            if micros := value.microsecond:
                funcname += "64"

            funcname += "BestEffort"

            args = [value.isoformat()]

            if micros % 1000:
                args.append(6)
            elif micros // 1000:
                args.append(3)

            if (timezone := dtype.timezone) is not None:
                args.append(timezone)

            return self.f[funcname](*args)
        elif dtype.is_date():
            return self.f.toDate(value.isoformat())
        elif dtype.is_array():
            value_type = dtype.value_type
            values = [
                self.visit_Literal(
                    ops.Literal(v, dtype=value_type), value=v, dtype=value_type
                )
                for v in value
            ]
            return self.f.array(*values)
        elif dtype.is_map():
            value_type = dtype.value_type
            keys = []
            values = []

            for k, v in value.items():
                keys.append(sge.convert(k))
                values.append(
                    self.visit_Literal(
                        ops.Literal(v, dtype=value_type),
                        value=v,
                        dtype=value_type,
                    )
                )

            return self.f.map(self.f.array(*keys), self.f.array(*values))
        elif dtype.is_struct():
            fields = [
                self.visit_Literal(
                    ops.Literal(v, dtype=field_type), value=v, dtype=field_type
                )
                for field_type, v in zip(dtype.types, value.values())
            ]
            return self.f.tuple(*fields)
        else:
            return None

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        if (unit := unit.short) in {"ms", "us", "ns"}:
            raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")
        return self.f.toDateTime(arg)

    def visit_TimestampTruncate(self, op, *, arg, unit):
        if (short := unit.short) == "W":
            funcname = "toMonday"
        else:
            funcname = f"toStartOf{unit.singular.capitalize()}"

        func = self.f[funcname]

        if short in ("Y", "Q", "M", "W", "D"):
            # these units return `Date` so we have to cast back to the
            # corresponding Ibis type
            return self.cast(func(arg), op.dtype)
        elif short in ("s", "ms", "us", "ns"):
            return func(self.f.toDateTime64(arg, op.arg.dtype.scale or 0))
        else:
            assert short in ("h", "m"), short
            return func(arg)

    visit_TimeTruncate = visit_TimestampTruncate

    def visit_DateTruncate(self, op, *, arg, unit):
        if unit.short == "D":
            # no op because truncating a date to a date has no effect
            return arg
        elif unit.short == "W":
            func = "toMonday"
        else:
            func = f"toStartOf{unit.singular.capitalize()}"

        # no cast needed here because all of the allowed units return `Date`
        # values
        return self.f[func](arg)

    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        if offset is not None:
            raise com.UnsupportedOperationError(
                "Timestamp bucket with offset is not supported"
            )

        return self.f.toStartOfInterval(arg, interval)

    def visit_DateFromYMD(self, op, *, year, month, day):
        return self.f.toDate(
            self.f.concat(
                self.f.toString(year),
                "-",
                self.f.leftPad(self.f.toString(month), 2, "0"),
                "-",
                self.f.leftPad(self.f.toString(day), 2, "0"),
            )
        )

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds, **_
    ):
        to_datetime = self.f.toDateTime(
            self.f.concat(
                self.f.toString(year),
                "-",
                self.f.leftPad(self.f.toString(month), 2, "0"),
                "-",
                self.f.leftPad(self.f.toString(day), 2, "0"),
                " ",
                self.f.leftPad(self.f.toString(hours), 2, "0"),
                ":",
                self.f.leftPad(self.f.toString(minutes), 2, "0"),
                ":",
                self.f.leftPad(self.f.toString(seconds), 2, "0"),
            )
        )
        if timezone := op.dtype.timezone:
            return self.f.toTimeZone(to_datetime, timezone)
        return to_datetime

    def visit_StringSplit(self, op, *, arg, delimiter):
        return self.f.splitByString(
            delimiter, self.cast(arg, dt.String(nullable=False))
        )

    def visit_GroupConcat(self, op, *, arg, sep, where, order_by):
        if order_by:
            raise com.UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                "not supported for this backend"
            )
        call = self.agg.groupArray(arg, where=where)
        return self.if_(self.f.empty(call), NULL, self.f.arrayStringConcat(call, sep))

    def visit_Cot(self, op, *, arg):
        return 1.0 / self.f.tan(arg)

    def visit_StructColumn(self, op, *, values, **_):
        return self.f.tuple(*values)

    def visit_Clip(self, op, *, arg, lower, upper):
        if upper is not None:
            arg = self.if_(self.f.isNull(arg), NULL, self.f.least(upper, arg))

        if lower is not None:
            arg = self.if_(self.f.isNull(arg), NULL, self.f.greatest(lower, arg))

        return arg

    def visit_StructField(self, op, *, arg, field: str):
        arg_dtype = op.arg.dtype
        idx = arg_dtype.names.index(field)
        return self.cast(sge.Dot(this=arg, expression=sge.convert(idx + 1)), op.dtype)

    def visit_Repeat(self, op, *, arg, times):
        return self.f.repeat(arg, self.f.accurateCast(times, "UInt64"))

    def visit_StringContains(self, op, haystack, needle):
        return self.f.position(haystack, needle) > 0

    def visit_Strip(self, op, *, arg):
        return sge.Trim(
            this=arg, position="BOTH", expression=sge.Literal.string(whitespace)
        )

    def visit_LPad(self, op, *, arg, length, pad):
        return self.f.leftPadUTF8(
            arg, self.f.greatest(self.f.lengthUTF8(arg), length), pad
        )

    def visit_RPad(self, op, *, arg, length, pad):
        return self.f.rightPadUTF8(
            arg, self.f.greatest(self.f.lengthUTF8(arg), length), pad
        )

    def visit_DayOfWeekIndex(self, op, *, arg):
        weekdays = len(calendar.day_name)
        return (((self.f.toDayOfWeek(arg) - 1) % weekdays) + weekdays) % weekdays

    def visit_DayOfWeekName(self, op, *, arg):
        # ClickHouse 20 doesn't support dateName
        #
        # ClickHouse 21 supports dateName is broken for regexen:
        # https://github.com/ClickHouse/ClickHouse/issues/32777
        #
        # ClickHouses 20 and 21 also have a broken case statement hence the ifnull:
        # https://github.com/ClickHouse/ClickHouse/issues/32849
        #
        # We test against 20 in CI, so we implement day_of_week_name as follows
        days = calendar.day_name
        num_weekdays = len(days)
        base = (
            ((self.f.toDayOfWeek(arg) - 1) % num_weekdays) + num_weekdays
        ) % num_weekdays
        return sge.Case(
            this=base,
            ifs=list(map(self.if_, *zip(*enumerate(days)))),
            default=sge.convert(""),
        )

    def visit_Map(self, op, *, keys, values):
        # cast here to allow lookups of nullable columns
        return self.cast(self.f.tuple(keys, values), op.dtype)

    def visit_MapGet(self, op, *, arg, key, default):
        return self.if_(self.f.mapContains(arg, key), arg[key], default)

    def visit_ArrayConcat(self, op, *, arg):
        return self.f.arrayConcat(*arg)

    def visit_BitAndOrXor(self, op, *, arg, where):
        if not (dtype := op.arg.dtype).is_unsigned_integer():
            nbits = dtype.nbytes * 8
            arg = self.f[f"reinterpretAsUInt{nbits}"](arg)
        return self.agg[f"group{type(op).__name__}"](arg, where=where)

    visit_BitAnd = visit_BitOr = visit_BitXor = visit_BitAndOrXor

    def visit_StandardDevVariance(self, op, *, how, where, **kw):
        funcs = {
            ops.StandardDev: "stddev",
            ops.Variance: "var",
            ops.Covariance: "covar",
        }
        func = funcs[type(op)]
        variants = {"sample": f"{func}Samp", "pop": f"{func}Pop"}
        funcname = variants[how]
        return self.agg[funcname](*kw.values(), where=where)

    visit_StandardDev = visit_Variance = visit_Covariance = visit_StandardDevVariance

    def visit_ArrayDistinct(self, op, *, arg):
        null_element = self.if_(
            self.f.countEqual(arg, NULL) > 0, self.f.array(NULL), self.f.array()
        )
        return self.f.arrayConcat(self.f.arrayDistinct(arg), null_element)

    def visit_ExtractMicrosecond(self, op, *, arg):
        dtype = op.dtype
        return self.cast(
            self.f.toUnixTimestamp64Micro(self.cast(arg, op.arg.dtype.copy(scale=6)))
            % 1_000_000,
            dtype,
        )

    def visit_ExtractMillisecond(self, op, *, arg):
        dtype = op.dtype
        return self.cast(
            self.f.toUnixTimestamp64Milli(self.cast(arg, op.arg.dtype.copy(scale=3)))
            % 1_000,
            dtype,
        )

    def visit_LagLead(self, op, *, arg, offset, default):
        args = [arg]

        if default is not None:
            if offset is None:
                offset = 1

            args.append(offset)
            args.append(default)
        elif offset is not None:
            args.append(offset)

        func = self.f[f"{type(op).__name__.lower()}InFrame"]
        return func(*args)

    visit_Lag = visit_Lead = visit_LagLead

    def visit_ExtractFile(self, op, *, arg):
        return self.f.cutFragment(self.f.pathFull(arg))

    def visit_ExtractQuery(self, op, *, arg, key):
        if key is not None:
            return self.f.extractURLParameter(arg, key)
        else:
            return self.f.queryString(arg)

    def visit_ArrayStringJoin(self, op, *, arg, sep):
        return self.f.arrayStringConcat(arg, sep)

    def visit_ArrayMap(self, op, *, arg, param, body, index):
        expressions = [param]
        args = [arg]

        if index is not None:
            expressions.append(index)
            args.append(self.f.range(0, self.f.length(arg)))

        func = sge.Lambda(this=body, expressions=expressions)

        return self.f.arrayMap(func, *args)

    def visit_ArrayFilter(self, op, *, arg, param, body, index):
        expressions = [param]
        args = [arg]

        if index is not None:
            expressions.append(index)
            args.append(self.f.range(0, self.f.length(arg)))

        func = sge.Lambda(this=body, expressions=expressions)

        return self.f.arrayFilter(func, *args)

    def visit_ArrayRemove(self, op, *, arg, other):
        x = sg.to_identifier(util.gen_name("x"))
        should_keep_null = sg.and_(x.is_(NULL), other.is_(sg.not_(NULL)))
        cond = sg.or_(x.neq(other), should_keep_null)
        return self.f.arrayFilter(sge.Lambda(this=cond, expressions=[x]), arg)

    def visit_ArrayUnion(self, op, *, left, right):
        arg = self.f.arrayConcat(left, right)
        null_element = self.if_(
            self.f.countEqual(arg, NULL) > 0, self.f.array(NULL), self.f.array()
        )
        return self.f.arrayConcat(self.f.arrayDistinct(arg), null_element)

    def visit_ArrayZip(self, op: ops.ArrayZip, *, arg, **_: Any) -> str:
        return self.f.arrayZip(*arg)

    def visit_ArrayCollect(self, op, *, arg, where, order_by, include_null, distinct):
        if include_null:
            raise com.UnsupportedOperationError(
                "`include_null=True` is not supported by the clickhouse backend"
            )
        func = self.agg.groupUniqArray if distinct else self.agg.groupArray
        return func(arg, where=where, order_by=order_by)

    def visit_First(self, op, *, arg, where, order_by, include_null):
        if include_null:
            raise com.UnsupportedOperationError(
                "`include_null=True` is not supported by the clickhouse backend"
            )
        return self.agg.any(arg, where=where, order_by=order_by)

    def visit_Last(self, op, *, arg, where, order_by, include_null):
        if include_null:
            raise com.UnsupportedOperationError(
                "`include_null=True` is not supported by the clickhouse backend"
            )
        return self.agg.anyLast(arg, where=where, order_by=order_by)

    def visit_ArgMin(self, op, *, arg, key, where):
        return sge.Dot(
            this=self.agg.argMin(self.f.tuple(arg), key, where=where),
            expression=sge.convert(1),
        )

    def visit_ArgMax(self, op, *, arg, key, where):
        return sge.Dot(
            this=self.agg.argMax(self.f.tuple(arg), key, where=where),
            expression=sge.convert(1),
        )

    def visit_CountDistinctStar(
        self, op: ops.CountDistinctStar, *, where, **_: Any
    ) -> str:
        columns = self.f.tuple(*map(sg.column, op.arg.schema.names))

        if where is not None:
            return self.f.countDistinctIf(columns, where)
        else:
            return self.f.countDistinct(columns)

    def visit_TimestampRange(self, op, *, start, stop, step):
        unit = op.step.dtype.unit.name.lower()

        if not isinstance(op.step, ops.Literal):
            raise com.UnsupportedOperationError(
                "ClickHouse doesn't support non-literal step values"
            )

        step_value = op.step.value

        offset = sg.to_identifier("offset")

        this = self.f.dateAdd(self.v[unit], offset, start)
        func = sge.Lambda(this=this, expressions=[offset])

        if step_value == 0:
            return self.f.array()

        return self.f.arrayMap(
            func, self.f.range(0, self.f.timestampDiff(unit, start, stop), step_value)
        )

    def visit_RegexSplit(self, op, *, arg, pattern):
        return self.f.splitByRegexp(pattern, self.cast(arg, dt.String(nullable=False)))

    @staticmethod
    def _generate_groups(groups):
        return groups

    def visit_DropColumns(self, op, *, parent, columns_to_drop):
        quoted = self.quoted
        excludes = [sg.column(column, quoted=quoted) for column in columns_to_drop]
        star = sge.Star(**{"except": excludes})
        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)
        column = sge.Column(this=star, table=table)
        return sg.select(column).from_(parent)

    def visit_TableUnnest(
        self,
        op,
        *,
        parent,
        column,
        column_name: str,
        offset: str | None,
        keep_empty: bool,
    ):
        quoted = self.quoted

        column_alias = sg.to_identifier(
            util.gen_name("table_unnest_column"), quoted=quoted
        )

        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)

        selcols = []

        overlaps_with_parent = column_name in op.parent.schema
        computed_column = column_alias.as_(column_name, quoted=quoted)

        if offset is not None:
            if overlaps_with_parent:
                selcols.append(
                    sge.Column(this=sge.Star(replace=[computed_column]), table=table)
                )
            else:
                selcols.append(sge.Column(this=STAR, table=table))
                selcols.append(computed_column)

            offset = sg.to_identifier(offset, quoted=quoted)
            selcols.append(offset)
        elif overlaps_with_parent:
            selcols.append(
                sge.Column(this=sge.Star(replace=[computed_column]), table=table)
            )
        else:
            selcols.append(sge.Column(this=STAR, table=table))
            selcols.append(computed_column)

        select = (
            sg.select(*selcols)
            .from_(parent)
            .join(
                sge.Join(
                    this=column.as_(column_alias, quoted=quoted),
                    kind="ARRAY",
                    side=None if not keep_empty else "LEFT",
                )
            )
        )

        if offset is not None:
            param = sg.to_identifier(util.gen_name("arr_enum"))
            func = sge.Lambda(this=param - 1, expressions=[param])
            return select.join(
                self.f.arrayMap(func, self.f.arrayEnumerate(column)).as_(offset)
            )

        return select

    def _cleanup_names(
        self, exprs: Mapping[str, sge.Expression]
    ) -> Iterator[sge.Expression]:
        """Compose `_gen_valid_name` and `_dedup_name` to clean up names in projections.

        ClickHouse has a bug where this fails to find the final `"o"."a"` column:

        ```sql
        SELECT
          "o"."a"
        FROM (
          SELECT
            "w"."a"
          FROM "t" AS "s"
          INNER JOIN "t" AS "w"
          USING ("a")
        ) AS "o"
        ```

        Adding a redundant aliasing operation (`"w"."a" AS "a"`) helps
        ClickHouse.
        """
        quoted = self.quoted
        return (
            value.as_(self._gen_valid_name(name), quoted=quoted, copy=False)
            for name, value in exprs.items()
        )

    def _array_reduction(self, arg):
        x = sg.to_identifier("x", quoted=self.quoted)
        not_null = sge.Lambda(this=x.is_(sg.not_(NULL)), expressions=[x])
        return self.f.arrayFilter(not_null, arg)

    def visit_ArrayMin(self, op, *, arg):
        return self.f.arrayReduce("min", self._array_reduction(arg))

    visit_ArrayAll = visit_ArrayMin

    def visit_ArrayMax(self, op, *, arg):
        return self.f.arrayReduce("max", self._array_reduction(arg))

    visit_ArrayAny = visit_ArrayMax

    def visit_ArraySum(self, op, *, arg):
        return self.f.arrayReduce("sum", self._array_reduction(arg))

    def visit_ArrayMean(self, op, *, arg):
        return self.f.arrayReduce("avg", self._array_reduction(arg))

    def _promote_bitshift_inputs(self, *, op, left, right):
        # clickhouse is incredibly pedantic about types allowed in bit shifting
        #
        # e.g., a UInt8 cannot be bitshift by more than 8 bits, UInt16 by more
        # than 16, and so on.
        #
        # This is why something like Ibis is necessary so that people have just
        # _consistent_ things, let alone *nice* things.
        left_dtype = op.left.dtype
        right_dtype = op.right.dtype

        if left_dtype != right_dtype:
            promoted = dt.higher_precedence(left_dtype, right_dtype)
            return self.cast(left, promoted), self.cast(right, promoted)
        return left, right

    def visit_BitwiseLeftShift(self, op, *, left, right):
        return self.f.bitShiftLeft(
            *self._promote_bitshift_inputs(op=op, left=left, right=right)
        )

    def visit_BitwiseRightShift(self, op, *, left, right):
        return self.f.bitShiftRight(
            *self._promote_bitshift_inputs(op=op, left=left, right=right)
        )

    def visit_MapContains(self, op, *, arg, key):
        return self.if_(
            sg.or_(arg.is_(NULL), key.is_(NULL)), NULL, self.f.mapContains(arg, key)
        )


compiler = ClickHouseCompiler()
