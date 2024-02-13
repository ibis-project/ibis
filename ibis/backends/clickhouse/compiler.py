from __future__ import annotations

import calendar
import math
from typing import Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sqlglot.compiler import (
    NULL,
    STAR,
    SQLGlotCompiler,
    parenthesize,
)
from ibis.backends.base.sqlglot.datatypes import ClickHouseType
from ibis.backends.base.sqlglot.dialects import ClickHouse
from ibis.backends.base.sqlglot.rewrites import rewrite_sample_as_filter


class ClickHouseCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = ClickHouse
    type_mapper = ClickHouseType
    rewrites = (rewrite_sample_as_filter, *SQLGlotCompiler.rewrites)

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.RowID,
            ops.CumeDist,
            ops.PercentRank,
            ops.Time,
            ops.TimeDelta,
            ops.StringToTimestamp,
            ops.Levenshtein,
        )
    )

    SIMPLE_OPS = {
        ops.All: "min",
        ops.Any: "max",
        ops.ApproxCountDistinct: "uniqHLL12",
        ops.ApproxMedian: "median",
        ops.ArgMax: "argMax",
        ops.ArgMin: "argMin",
        ops.ArrayCollect: "groupArray",
        ops.ArrayContains: "has",
        ops.ArrayFlatten: "arrayFlatten",
        ops.ArrayIntersect: "arrayIntersect",
        ops.ArrayPosition: "indexOf",
        ops.BitwiseAnd: "bitAnd",
        ops.BitwiseLeftShift: "bitShiftLeft",
        ops.BitwiseNot: "bitNot",
        ops.BitwiseOr: "bitOr",
        ops.BitwiseRightShift: "bitShiftRight",
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
        ops.First: "any",
        ops.IntegerRange: "range",
        ops.IsInf: "isInfinite",
        ops.IsNan: "isNaN",
        ops.IsNull: "isNull",
        ops.LStrip: "trimLeft",
        ops.Last: "anyLast",
        ops.Ln: "log",
        ops.Log10: "log10",
        ops.MapContains: "mapContains",
        ops.MapKeys: "mapKeys",
        ops.MapLength: "length",
        ops.MapMerge: "mapUpdate",
        ops.MapValues: "mapValues",
        ops.Median: "quantileExactExclusive",
        ops.NotNull: "isNotNull",
        ops.NullIf: "nullIf",
        ops.RStrip: "trimRight",
        ops.RandomScalar: "randCanonical",
        ops.RegexReplace: "replaceRegexpAll",
        ops.RowNumber: "row_number",
        ops.StartsWith: "startsWith",
        ops.StrRight: "right",
        ops.Strftime: "formatDateTime",
        ops.StringLength: "length",
        ops.StringReplace: "replaceAll",
        ops.Strip: "trimBoth",
        ops.TimestampNow: "now",
        ops.TypeOf: "toTypeName",
        ops.Unnest: "arrayJoin",
    }

    def _aggregate(self, funcname: str, *args, where):
        has_filter = where is not None
        func = self.f[funcname + "If" * has_filter]
        args += (where,) * has_filter

        return func(*args, dialect=self.dialect)

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
        start = parenthesize(op.start, start)
        start_correct = self.if_(start < 0, start, start + 1)

        if stop is not None:
            stop = parenthesize(op.stop, stop)

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

    def visit_Quantile(self, op, *, arg, quantile, where):
        if where is None:
            return self.agg.quantile(arg, quantile, where=where)

        func = "quantile" + "s" * isinstance(op, ops.MultiQuantile)
        return sge.ParameterizedAgg(
            this=f"{func}If",
            expressions=util.promote_list(quantile),
            params=[arg, where],
        )

    visit_MultiQuantile = visit_Quantile

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "pop":
            raise ValueError(
                "ClickHouse only implements `sample` correlation coefficient"
            )
        return self.agg.corr(left, right, where=where)

    def visit_Arbitrary(self, op, *, arg, how, where):
        if how == "first":
            return self.agg.any(arg, where=where)
        elif how == "last":
            return self.agg.anyLast(arg, where=where)
        else:
            assert how == "heavy"
            return self.agg.anyHeavy(arg, where=where)

    def visit_Substring(self, op, *, arg, start, length):
        # Clickhouse is 1-indexed
        suffix = (length,) * (length is not None)
        if_pos = self.f.substring(arg, start + 1, *suffix)
        if_neg = self.f.substring(arg, self.f.length(arg) + start + 1, *suffix)
        return self.if_(start >= 0, if_pos, if_neg)

    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise com.UnsupportedOperationError(
                "String find doesn't support end argument"
            )

        if start is not None:
            return self.f.locate(arg, substr, start)

        return self.f.locate(arg, substr)

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
        return self.f.sipHash64(arg)

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
        elif dtype.is_string():
            return sge.convert(str(value).replace(r"\0", r"\\0"))
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
        converters = {
            "Y": "toStartOfYear",
            "M": "toStartOfMonth",
            "W": "toMonday",
            "D": "toDate",
            "h": "toStartOfHour",
            "m": "toStartOfMinute",
            "s": "toDateTime",
        }

        unit = unit.short
        if (converter := converters.get(unit)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")

        return self.f[converter](arg)

    visit_TimeTruncate = visit_DateTruncate = visit_TimestampTruncate

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

    def visit_GroupConcat(self, op, *, arg, sep, where):
        call = self.agg.groupArray(arg, where=where)
        return self.if_(self.f.empty(call), NULL, self.f.arrayStringConcat(call, sep))

    def visit_Cot(self, op, *, arg):
        return 1.0 / self.f.tan(arg)

    def visit_StructColumn(self, op, *, values, names):
        # ClickHouse struct types cannot be nullable
        # (non-nested fields can be nullable)
        return self.cast(self.f.tuple(*values), op.dtype.copy(nullable=False))

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
        return self.f.locate(haystack, needle) > 0

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

    def visit_ArrayMap(self, op, *, arg, param, body):
        func = sge.Lambda(this=body, expressions=[param])
        return self.f.arrayMap(func, arg)

    def visit_ArrayFilter(self, op, *, arg, param, body):
        func = sge.Lambda(this=body, expressions=[param])
        return self.f.arrayFilter(func, arg)

    def visit_ArrayRemove(self, op, *, arg, other):
        x = sg.to_identifier("x")
        body = x.neq(other)
        return self.f.arrayFilter(sge.Lambda(this=body, expressions=[x]), arg)

    def visit_ArrayUnion(self, op, *, left, right):
        arg = self.f.arrayConcat(left, right)
        null_element = self.if_(
            self.f.countEqual(arg, NULL) > 0, self.f.array(NULL), self.f.array()
        )
        return self.f.arrayConcat(self.f.arrayDistinct(arg), null_element)

    def visit_ArrayZip(self, op: ops.ArrayZip, *, arg, **_: Any) -> str:
        return self.f.arrayZip(*arg)

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

        func = sge.Lambda(
            this=self.f.dateAdd(sg.to_identifier(unit), offset, start),
            expressions=[offset],
        )

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
