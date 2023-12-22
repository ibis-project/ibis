from __future__ import annotations

import calendar
import math
from functools import singledispatchmethod
from typing import Any

import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot import exp
from sqlglot.dialects import ClickHouse
from sqlglot.dialects.dialect import rename_func

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
from ibis.backends.clickhouse.datatypes import ClickhouseType
from ibis.expr.rewrites import rewrite_sample

ClickHouse.Generator.TRANSFORMS |= {
    exp.ArraySize: rename_func("length"),
    exp.ArraySort: rename_func("arraySort"),
}


class ClickHouseCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "clickhouse"
    type_mapper = ClickhouseType
    rewrites = (rewrite_sample, *SQLGlotCompiler.rewrites)

    def _aggregate(self, funcname: str, *args, where):
        has_filter = where is not None
        func = self.f[funcname + "If" * has_filter]
        args += (where,) * has_filter
        return func(*args)

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.Cast)
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

    @visit_node.register(ops.TryCast)
    def visit_TryCast(self, op, *, arg, to):
        return self.f.accurateCastOrNull(arg, self.type_mapper.to_string(to))

    @visit_node.register(ops.ArrayIndex)
    def visit_ArrayIndex(self, op, *, arg, index):
        return arg[self.if_(index >= 0, index + 1, index)]

    @visit_node.register(ops.ArrayRepeat)
    def visit_ArrayRepeat(self, op, *, arg, times):
        param = sg.to_identifier("_")
        func = sge.Lambda(this=arg, expressions=[param])
        return self.f.arrayFlatten(self.f.arrayMap(func, self.f.range(times)))

    @visit_node.register(ops.ArraySlice)
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

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, where, arg):
        if where is not None:
            return self.f.countIf(where)
        return sge.Count(this=STAR)

    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.MultiQuantile)
    def visit_QuantileMultiQuantile(self, op, *, arg, quantile, where):
        if where is None:
            return self.agg.quantile(arg, quantile, where=where)

        func = "quantile" + "s" * isinstance(op, ops.MultiQuantile)
        return sge.ParameterizedAgg(
            this=f"{func}If",
            expressions=util.promote_list(quantile),
            params=[arg, where],
        )

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "pop":
            raise ValueError(
                "ClickHouse only implements `sample` correlation coefficient"
            )
        return self.agg.corr(left, right, where=where)

    @visit_node.register(ops.Arbitrary)
    def visit_Arbitrary(self, op, *, arg, how, where):
        if how == "first":
            return self.agg.any(arg, where=where)
        elif how == "last":
            return self.agg.anyLast(arg, where=where)
        else:
            assert how == "heavy"
            return self.agg.anyHeavy(arg, where=where)

    @visit_node.register(ops.Substring)
    def visit_Substring(self, op, *, arg, start, length):
        # Clickhouse is 1-indexed
        suffix = (length,) * (length is not None)
        if_pos = self.f.substring(arg, start + 1, *suffix)
        if_neg = self.f.substring(arg, self.f.length(arg) + start + 1, *suffix)
        return self.if_(start >= 0, if_pos, if_neg)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise com.UnsupportedOperationError(
                "String find doesn't support end argument"
            )

        if start is not None:
            return self.f.locate(arg, substr, start)

        return self.f.locate(arg, substr)

    @visit_node.register(ops.RegexSearch)
    def visit_RegexSearch(self, op, *, arg, pattern):
        return sge.RegexpLike(this=arg, expression=pattern)

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        arg = self.cast(arg, dt.String(nullable=False))

        pattern = self.f.concat("(", pattern, ")")

        if index is None:
            index = 0

        index += 1

        then = self.f.extractGroups(arg, pattern)[index]

        return self.if_(self.f.notEmpty(then), then, NULL)

    @visit_node.register(ops.FindInSet)
    def visit_FindInSet(self, op, *, needle, values):
        return self.f.indexOf(self.f.array(*values), needle)

    @visit_node.register(ops.Sign)
    def visit_Sign(self, op, *, arg):
        """Workaround for missing sign function in older versions of clickhouse."""
        return self.f.intDivOrZero(arg, self.f.abs(arg))

    @visit_node.register(ops.Hash)
    def visit_Hash(self, op, *, arg):
        return self.f.sipHash64(arg)

    @visit_node.register(ops.HashBytes)
    def visit_HashBytes(self, op, *, arg, how):
        supported_algorithms = frozenset(
            (
                "MD5",
                "halfMD5",
                "SHA1",
                "SHA224",
                "SHA256",
                "intHash32",
                "intHash64",
                "cityHash64",
                "sipHash64",
                "sipHash128",
            )
        )
        if how not in supported_algorithms:
            raise com.UnsupportedOperationError(f"Unsupported hash algorithm {how}")

        return self.f[how](arg)

    @visit_node.register(ops.IntervalFromInteger)
    def visit_IntervalFromInteger(self, op, *, arg, unit):
        dtype = op.dtype
        if dtype.unit.short in ("ms", "us", "ns"):
            raise com.UnsupportedOperationError(
                "Clickhouse doesn't support subsecond interval resolutions"
            )
        return super().visit_node(op, arg=arg, unit=unit)

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype, **kw):
        if value is None:
            return super().visit_node(op, value=value, dtype=dtype, **kw)
        elif dtype.is_inet():
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
                    ops.Literal(v, dtype=value_type), value=v, dtype=value_type, **kw
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
                        **kw,
                    )
                )

            return self.f.map(self.f.array(*keys), self.f.array(*values))
        elif dtype.is_struct():
            fields = [
                self.visit_Literal(
                    ops.Literal(v, dtype=field_type), value=v, dtype=field_type, **kw
                )
                for field_type, v in zip(dtype.types, value.values())
            ]
            return self.f.tuple(*fields)
        else:
            return super().visit_node(op, value=value, dtype=dtype, **kw)

    @visit_node.register(ops.TimestampFromUNIX)
    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        if (unit := unit.short) in {"ms", "us", "ns"}:
            raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")
        return self.f.toDateTime(arg)

    @visit_node.register(ops.DateTruncate)
    @visit_node.register(ops.TimestampTruncate)
    @visit_node.register(ops.TimeTruncate)
    def visit_TimeTruncate(self, op, *, arg, unit):
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

    @visit_node.register(ops.TimestampBucket)
    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        if offset is not None:
            raise com.UnsupportedOperationError(
                "Timestamp bucket with offset is not supported"
            )

        return self.f.toStartOfInterval(arg, interval)

    @visit_node.register(ops.DateFromYMD)
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

    @visit_node.register(ops.TimestampFromYMDHMS)
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

    @visit_node.register(ops.StringSplit)
    def visit_StringSplit(self, op, *, arg, delimiter):
        return self.f.splitByString(
            delimiter, self.cast(arg, dt.String(nullable=False))
        )

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, sep, arg):
        return self.f.arrayStringConcat(self.f.array(*arg), sep)

    @visit_node.register(ops.Capitalize)
    def visit_Capitalize(self, op, *, arg):
        return self.f.concat(
            self.f.upper(self.f.substr(arg, 1, 1)), self.f.lower(self.f.substr(arg, 2))
        )

    @visit_node.register(ops.GroupConcat)
    def visit_GroupConcat(self, op, *, arg, sep, where):
        call = self.agg.groupArray(arg, where=where)
        return self.if_(self.f.empty(call), NULL, self.f.arrayStringConcat(call, sep))

    @visit_node.register(ops.Cot)
    def visit_Cot(self, op, *, arg):
        return 1.0 / self.f.tan(arg)

    @visit_node.register(ops.StructColumn)
    def visit_StructColumn(self, op, *, values, names):
        # ClickHouse struct types cannot be nullable
        # (non-nested fields can be nullable)
        return self.cast(self.f.tuple(*values), op.dtype.copy(nullable=False))

    @visit_node.register(ops.Clip)
    def visit_Clip(self, op, *, arg, lower, upper):
        if upper is not None:
            arg = self.if_(self.f.isNull(arg), NULL, self.f.least(upper, arg))

        if lower is not None:
            arg = self.if_(self.f.isNull(arg), NULL, self.f.greatest(lower, arg))

        return arg

    @visit_node.register(ops.StructField)
    def visit_StructField(self, op, *, arg, field: str):
        arg_dtype = op.arg.dtype
        idx = arg_dtype.names.index(field)
        return self.cast(sge.Dot(this=arg, expression=sge.convert(idx + 1)), op.dtype)

    @visit_node.register(ops.Repeat)
    def visit_Repeat(self, op, *, arg, times):
        return self.f.repeat(arg, self.f.accurateCast(times, "UInt64"))

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, haystack, needle):
        return self.f.locate(haystack, needle) > 0

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        weekdays = len(calendar.day_name)
        return (((self.f.toDayOfWeek(arg) - 1) % weekdays) + weekdays) % weekdays

    @visit_node.register(ops.DayOfWeekName)
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

    @visit_node.register(ops.Map)
    def visit_Map(self, op, *, keys, values):
        # cast here to allow lookups of nullable columns
        return self.cast(self.f.tuple(keys, values), op.dtype)

    @visit_node.register(ops.MapGet)
    def visit_MapGet(self, op, *, arg, key, default):
        return self.if_(self.f.mapContains(arg, key), arg[key], default)

    @visit_node.register(ops.ArrayConcat)
    def visit_ArrayConcat(self, op, *, arg):
        return self.f.arrayConcat(*arg)

    @visit_node.register(ops.BitAnd)
    @visit_node.register(ops.BitOr)
    @visit_node.register(ops.BitXor)
    def visit_BitAndOrXor(self, op, *, arg, where):
        if not (dtype := op.arg.dtype).is_unsigned_integer():
            nbits = dtype.nbytes * 8
            arg = self.f[f"reinterpretAsUInt{nbits}"](arg)
        return self.agg[f"group{type(op).__name__}"](arg, where=where)

    @visit_node.register(ops.StandardDev)
    @visit_node.register(ops.Variance)
    @visit_node.register(ops.Covariance)
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

    @visit_node.register(ops.ArrayDistinct)
    def visit_ArrayDistinct(self, op, *, arg):
        null_element = self.if_(
            self.f.countEqual(arg, NULL) > 0, self.f.array(NULL), self.f.array()
        )
        return self.f.arrayConcat(self.f.arrayDistinct(arg), null_element)

    @visit_node.register(ops.ExtractMicrosecond)
    def visit_ExtractMicrosecond(self, op, *, arg):
        dtype = op.dtype
        return self.cast(
            self.f.toUnixTimestamp64Micro(self.cast(arg, op.arg.dtype.copy(scale=6)))
            % 1_000_000,
            dtype,
        )

    @visit_node.register(ops.ExtractMillisecond)
    def visit_ExtractMillisecond(self, op, *, arg):
        dtype = op.dtype
        return self.cast(
            self.f.toUnixTimestamp64Milli(self.cast(arg, op.arg.dtype.copy(scale=3)))
            % 1_000,
            dtype,
        )

    @visit_node.register(ops.Lag)
    @visit_node.register(ops.Lead)
    def formatter(self, op, *, arg, offset, default):
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

    @visit_node.register(ops.ExtractFile)
    def visit_ExtractFile(self, op, *, arg):
        return self.f.cutFragment(self.f.pathFull(arg))

    @visit_node.register(ops.ExtractQuery)
    def visit_ExtractQuery(self, op, *, arg, key):
        if key is not None:
            return self.f.extractURLParameter(arg, key)
        else:
            return self.f.queryString(arg)

    @visit_node.register(ops.ArrayStringJoin)
    def visit_ArrayStringJoin(self, op, *, arg, sep):
        return self.f.arrayStringConcat(arg, sep)

    @visit_node.register(ops.ArrayMap)
    def visit_ArrayMap(self, op, *, arg, param, body):
        func = sge.Lambda(this=body, expressions=[param])
        return self.f.arrayMap(func, arg)

    @visit_node.register(ops.ArrayFilter)
    def visit_ArrayFilter(self, op, *, arg, param, body):
        func = sge.Lambda(this=body, expressions=[param])
        return self.f.arrayFilter(func, arg)

    @visit_node.register(ops.ArrayRemove)
    def visit_ArrayRemove(self, op, *, arg, other):
        x = sg.to_identifier("x")
        body = x.neq(other)
        return self.f.arrayFilter(sge.Lambda(this=body, expressions=[x]), arg)

    @visit_node.register(ops.ArrayUnion)
    def visit_ArrayUnion(self, op, *, left, right):
        arg = self.f.arrayConcat(left, right)
        null_element = self.if_(
            self.f.countEqual(arg, NULL) > 0, self.f.array(NULL), self.f.array()
        )
        return self.f.arrayConcat(self.f.arrayDistinct(arg), null_element)

    @visit_node.register(ops.ArrayZip)
    def visit_ArrayZip(self, op: ops.ArrayZip, *, arg, **_: Any) -> str:
        return self.f.arrayZip(*arg)

    @visit_node.register(ops.CountDistinctStar)
    def visit_CountDistinctStar(
        self, op: ops.CountDistinctStar, *, where, **_: Any
    ) -> str:
        columns = self.f.tuple(*map(sg.column, op.arg.schema.names))

        if where is not None:
            return self.f.countDistinctIf(columns, where)
        else:
            return self.f.countDistinct(columns)

    @visit_node.register(ops.TimestampRange)
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

    @visit_node.register(ops.RegexSplit)
    def visit_RegexSplit(self, op, *, arg, pattern):
        return self.f.splitByRegexp(pattern, self.cast(arg, dt.String(nullable=False)))

    @staticmethod
    def _generate_groups(groups):
        return groups

    @visit_node.register(ops.RowID)
    @visit_node.register(ops.CumeDist)
    @visit_node.register(ops.PercentRank)
    @visit_node.register(ops.Time)
    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.StringToTimestamp)
    @visit_node.register(ops.Levenshtein)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
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
    ops.LPad: "leftPad",
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
    ops.RPad: "rightPad",
    ops.RStrip: "trimRight",
    ops.RandomScalar: "randCanonical",
    ops.RegexReplace: "replaceRegexpAll",
    ops.Repeat: "repeat",
    ops.RowNumber: "row_number",
    ops.StartsWith: "startsWith",
    ops.StrRight: "right",
    ops.Strftime: "formatDateTime",
    ops.StringAscii: "ascii",
    ops.StringLength: "length",
    ops.StringReplace: "replaceAll",
    ops.Strip: "trimBoth",
    ops.TimestampNow: "now",
    ops.Translate: "translate",
    ops.TypeOf: "toTypeName",
    ops.Unnest: "arrayJoin",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @ClickHouseCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @ClickHouseCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(ClickHouseCompiler, f"visit_{_op.__name__}", _fmt)

del _op, _name, _fmt
