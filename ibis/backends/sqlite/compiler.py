from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public
from sqlglot.dialects.sqlite import SQLite

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import SQLiteType
from ibis.backends.base.sqlglot.rewrites import (
    rewrite_first_to_first_value,
    rewrite_last_to_last_value,
)
from ibis.common.temporal import DateUnit, IntervalUnit
from ibis.expr.rewrites import rewrite_sample

SQLite.Generator.TYPE_MAPPING |= {
    sge.DataType.Type.BOOLEAN: "BOOLEAN",
}


@public
class SQLiteCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "sqlite"
    quoted = True
    type_mapper = SQLiteType
    rewrites = SQLGlotCompiler.rewrites + (
        rewrite_sample,
        rewrite_first_to_first_value,
        rewrite_last_to_last_value,
    )

    NAN = sge.NULL
    POS_INF = sge.Literal.number("1e999")
    NEG_INF = sge.Literal.number("-1e999")

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.Levenshtein)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.IsNan)
    @visit_node.register(ops.IsInf)
    @visit_node.register(ops.Covariance)
    @visit_node.register(ops.Correlation)
    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.Median)
    @visit_node.register(ops.ApproxMedian)
    @visit_node.register(ops.Array)
    @visit_node.register(ops.ArrayConcat)
    @visit_node.register(ops.ArrayStringJoin)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.ArrayContains)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayLength)
    @visit_node.register(ops.ArraySort)
    @visit_node.register(ops.ArrayStringJoin)
    @visit_node.register(ops.CountDistinctStar)
    @visit_node.register(ops.IntervalBinary)
    @visit_node.register(ops.IntervalAdd)
    @visit_node.register(ops.IntervalSubtract)
    @visit_node.register(ops.IntervalMultiply)
    @visit_node.register(ops.IntervalFloorDivide)
    @visit_node.register(ops.IntervalFromInteger)
    @visit_node.register(ops.TimestampBucket)
    @visit_node.register(ops.TimestampAdd)
    @visit_node.register(ops.TimestampSub)
    @visit_node.register(ops.TimestampDiff)
    @visit_node.register(ops.StringToTimestamp)
    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.TimestampDelta)
    @visit_node.register(ops.TryCast)
    def visit_Undefined(self, op, **kwargs):
        return super().visit_Undefined(op, **kwargs)

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to) -> sge.Cast:
        if to.is_timestamp():
            if to.timezone not in (None, "UTC"):
                raise com.UnsupportedOperationError(
                    "SQLite does not support casting to timezones other than 'UTC'"
                )
            if op.arg.dtype.is_numeric():
                return self.f.datetime(arg, "unixepoch")
            else:
                return self.f.strftime("%Y-%m-%d %H:%M:%f", arg)
        elif to.is_date():
            return self.f.date(arg)
        elif to.is_time():
            return self.f.time(arg)
        return super().visit_Cast(op, arg=arg, to=to)

    @visit_node.register(ops.Limit)
    def visit_Limit(self, op, *, parent, n, offset):
        # SQLite doesn't support compiling an OFFSET without a LIMIT, but
        # treats LIMIT -1 as no limit
        return super().visit_Limit(
            op, parent=parent, n=(-1 if n is None else n), offset=offset
        )

    @visit_node.register(ops.WindowBoundary)
    def visit_WindowBoundary(self, op, *, value, preceding):
        if op.value.dtype.is_interval():
            raise com.OperationNotDefinedError(
                "Interval window bounds not supported by SQLite"
            )
        return super().visit_WindowBoundary(op, value=value, preceding=preceding)

    @visit_node.register(ops.JoinLink)
    def visit_JoinLink(self, op, **kwargs):
        if op.how == "asof":
            raise com.UnsupportedOperationError(
                "ASOF joins are not supported by SQLite"
            )
        return super().visit_JoinLink(op, **kwargs)

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        return arg.like(self.f.concat(start, "%"))

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return arg.like(self.f.concat("%", end))

    @visit_node.register(ops.StrRight)
    def visit_StrRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -nchars, nchars)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if op.end is not None:
            raise NotImplementedError("`end` not yet implemented")

        if op.start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.instr(arg, substr)
            return sg.case().when(pos > 0, pos + start).else_(0)

        return self.f.instr(arg, substr)

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep):
        args = [arg[0]]
        for item in arg[1:]:
            args.extend([sep, item])
        return self.f.concat(*args)

    @visit_node.register(ops.StringContains)
    def visit_Contains(self, op, *, haystack, needle):
        return self.f.instr(haystack, needle) >= 1

    @visit_node.register(ops.ExtractQuery)
    def visit_ExtractQuery(self, op, *, arg, key):
        if op.key is None:
            return self.f._ibis_extract_full_query(arg)
        return self.f._ibis_extract_query(arg, key)

    @visit_node.register(ops.Greatest)
    def visit_Greatest(self, op, *, arg):
        return self.f.max(*arg)

    @visit_node.register(ops.Least)
    def visit_Least(self, op, *, arg):
        return self.f.min(*arg)

    @visit_node.register(ops.IdenticalTo)
    def visit_IdenticalTo(self, op, *, left, right):
        return sge.Is(this=left, expression=right)

    @visit_node.register(ops.Clip)
    def visit_Clip(self, op, *, arg, lower, upper):
        if upper is not None:
            arg = self.if_(arg.is_(sge.NULL), arg, self.f.min(upper, arg))

        if lower is not None:
            arg = self.if_(arg.is_(sge.NULL), arg, self.f.max(lower, arg))

        return arg

    @visit_node.register(ops.RandomScalar)
    def visit_RandomScalar(self, op):
        return 0.5 + self.f.random() / sge.Literal.number(float(-1 << 64))

    @visit_node.register(ops.Cot)
    def visit_Cot(self, op, *, arg):
        return 1 / self.f.tan(arg)

    @visit_node.register(ops.Arbitrary)
    def visit_Arbitrary(self, op, *, arg, how, where):
        if op.how == "heavy":
            raise com.OperationNotDefinedError(
                "how='heavy' not implemented for the SQLite backend"
            )

        return self._aggregate(f"_ibis_arbitrary_{how}", arg, where=where)

    @visit_node.register(ops.ArgMin)
    def visit_ArgMin(self, *args, **kwargs):
        return self._visit_arg_reduction("min", *args, **kwargs)

    @visit_node.register(ops.ArgMax)
    def visit_ArgMax(self, *args, **kwargs):
        return self._visit_arg_reduction("max", *args, **kwargs)

    def _visit_arg_reduction(self, func, op, *, arg, key, where):
        cond = arg.is_(sg.not_(sge.NULL))

        if op.where is not None:
            cond = sg.and_(cond, where)

        agg = self._aggregate(func, key, where=cond)
        return self.f.anon.json_extract(self.f.json_array(arg, agg), "$[0]")

    @visit_node.register(ops.Variance)
    def visit_Variance(self, op, *, arg, how, where):
        return self._aggregate(f"_ibis_var_{op.how}", arg, where=where)

    @visit_node.register(ops.StandardDev)
    def visit_StandardDev(self, op, *, arg, how, where):
        var = self._aggregate(f"_ibis_var_{op.how}", arg, where=where)
        return self.f.sqrt(var)

    @visit_node.register(ops.ApproxCountDistinct)
    def visit_ApproxCountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    @visit_node.register(ops.Strftime)
    def visit_Strftime(self, op, *, arg, format_str):
        return self.f.strftime(format_str, arg)

    @visit_node.register(ops.DateFromYMD)
    def visit_DateFromYMD(self, op, *, year, month, day):
        return self.f.date(self.f.printf("%04d-%02d-%02d", year, month, day))

    @visit_node.register(ops.TimeFromHMS)
    def visit_TimeFromHMS(self, op, *, hours, minutes, seconds):
        return self.f.time(self.f.printf("%02d:%02d:%02d", hours, minutes, seconds))

    @visit_node.register(ops.TimestampFromYMDHMS)
    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        return self.f.datetime(
            self.f.printf(
                "%04d-%02d-%02d %02d:%02d:%02d%s",
                year,
                month,
                day,
                hours,
                minutes,
                seconds,
            )
        )

    def _temporal_truncate(self, func, arg, unit):
        modifiers = {
            DateUnit.DAY: ("start of day",),
            DateUnit.WEEK: ("weekday 0", "-6 days"),
            DateUnit.MONTH: ("start of month",),
            DateUnit.YEAR: ("start of year",),
            IntervalUnit.DAY: ("start of day",),
            IntervalUnit.WEEK: ("weekday 0", "-6 days", "start of day"),
            IntervalUnit.MONTH: ("start of month",),
            IntervalUnit.YEAR: ("start of year",),
        }

        params = modifiers.get(unit)
        if params is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit}")
        return func(arg, *params)

    @visit_node.register(ops.DateTruncate)
    def visit_DateTruncate(self, op, *, arg, unit):
        return self._temporal_truncate(self.f.date, arg, unit)

    @visit_node.register(ops.TimestampTruncate)
    def visit_TimestampTruncate(self, op, *, arg, unit):
        return self._temporal_truncate(self.f.datetime, arg, unit)

    @visit_node.register(ops.DateAdd)
    @visit_node.register(ops.DateSub)
    def visit_DateArithmetic(self, op, *, left, right):
        unit = op.right.dtype.unit
        sign = "+" if isinstance(op, ops.DateAdd) else "-"
        if unit not in (IntervalUnit.YEAR, IntervalUnit.MONTH, IntervalUnit.DAY):
            raise com.UnsupportedOperationError(
                "SQLite does not allow binary op {sign!r} with INTERVAL offset {unit}"
            )
        if isinstance(op.right, ops.Literal):
            return self.f.date(left, f"{sign}{op.right.value} {unit.plural}")
        else:
            return self.f.date(left, self.f.concat(sign, right, f" {unit.plural}"))

    @visit_node.register(ops.DateDiff)
    def visit_DateDiff(self, op, *, left, right):
        return self.f.julianday(left) - self.f.julianday(right)

    @visit_node.register(ops.ExtractYear)
    def visit_ExtractYear(self, op, *, arg):
        return self.cast(self.f.strftime("%Y", arg), dt.int64)

    @visit_node.register(ops.ExtractQuarter)
    def visit_ExtractQuarter(self, op, *, arg):
        return (self.f.strftime("%m", arg) + 2) / 3

    @visit_node.register(ops.ExtractMonth)
    def visit_ExtractMonth(self, op, *, arg):
        return self.cast(self.f.strftime("%m", arg), dt.int64)

    @visit_node.register(ops.ExtractDay)
    def visit_ExtractDay(self, op, *, arg):
        return self.cast(self.f.strftime("%d", arg), dt.int64)

    @visit_node.register(ops.ExtractDayOfYear)
    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.cast(self.f.strftime("%j", arg), dt.int64)

    @visit_node.register(ops.ExtractHour)
    def visit_ExtractHour(self, op, *, arg):
        return self.cast(self.f.strftime("%H", arg), dt.int64)

    @visit_node.register(ops.ExtractMinute)
    def visit_ExtractMinute(self, op, *, arg):
        return self.cast(self.f.strftime("%M", arg), dt.int64)

    @visit_node.register(ops.ExtractSecond)
    def visit_ExtractSecond(self, op, *, arg):
        return self.cast(self.f.strftime("%S", arg), dt.int64)

    @visit_node.register(ops.ExtractMillisecond)
    def visit_Millisecond(self, op, *, arg):
        return self.cast(self.f.mod(self.f.strftime("%f", arg) * 1000, 1000), dt.int64)

    @visit_node.register(ops.ExtractMicrosecond)
    def visit_Microsecond(self, op, *, arg):
        return self.cast(
            self.f.mod(self.cast(self.f.strftime("%f", arg), dt.int64), 1000), dt.int64
        )

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg):
        """ISO week of year.

        This solution is based on https://stackoverflow.com/a/15511864 and handle
        the edge cases when computing ISO week from non-ISO week.

        The implementation gives the same results as `datetime.isocalendar()`.

        The year's week that "wins" the day is the year with more allotted days.

        For example:

        ```
        $ cal '2011-01-01'
            January 2011
        Su Mo Tu We Th Fr Sa
                        |1|
        2  3  4  5  6  7  8
        9 10 11 12 13 14 15
        16 17 18 19 20 21 22
        23 24 25 26 27 28 29
        30 31
        ```

        Here the ISO week number is `52` since the day occurs in a week with more
        days in the week occurring in the _previous_ week's year.

        ```
        $ cal '2012-12-31'
            December 2012
        Su Mo Tu We Th Fr Sa
                        1
        2  3  4  5  6  7  8
        9 10 11 12 13 14 15
        16 17 18 19 20 21 22
        23 24 25 26 27 28 29
        30 |31|
        ```

        Here the ISO week of year is `1` since the day occurs in a week with more
        days in the week occurring in the _next_ week's year.
        """
        date = self.f.date(arg, "-3 days", "weekday 4")
        return (self.f.strftime("%j", date) - 1) / 7 + 1

    @visit_node.register(ops.ExtractEpochSeconds)
    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.cast((self.f.julianday(arg) - 2440587.5) * 86400.0, dt.int64)

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.cast(
            self.f.mod(self.cast(self.f.strftime("%w", arg) + 6, dt.int64), 7), dt.int64
        )

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg):
        return sge.Case(
            this=self.f.strftime("%w", arg),
            ifs=[
                self.if_("0", "Sunday"),
                self.if_("1", "Monday"),
                self.if_("2", "Tuesday"),
                self.if_("3", "Wednesday"),
                self.if_("4", "Thursday"),
                self.if_("5", "Friday"),
                self.if_("6", "Saturday"),
            ],
        )

    @visit_node.register(ops.Xor)
    def visit_Xor(self, op, *, left, right):
        return (left.or_(right)).and_(sg.not_(left.and_(right)))

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_binary():
            return self.f.unhex(value.hex())

        if dtype.is_decimal():
            value = float(value)
            dtype = dt.double(nullable=dtype.nullable)
        elif dtype.is_uuid():
            value = str(value)
            dtype = dt.string(nullable=dtype.nullable)
        elif dtype.is_interval():
            value = int(value)
            dtype = dt.int64(nullable=dtype.nullable)
        elif dtype.is_date() or dtype.is_timestamp() or dtype.is_time():
            # To ensure comparisons apply uniformly between temporal values
            # (which are always represented as strings), we need to enforce
            # that temporal literals are formatted the same way that SQLite
            # formats them. This means " " instead of "T" and no offset suffix
            # for UTC.
            value = (
                value.isoformat()
                .replace("T", " ")
                .replace("Z", "")
                .replace("+00:00", "")
            )
            dtype = dt.string(nullable=dtype.nullable)
        elif (
            dtype.is_map()
            or dtype.is_struct()
            or dtype.is_array()
            or dtype.is_geospatial()
        ):
            raise com.UnsupportedBackendType(f"Unsupported type: {dtype!r}")
        return super().visit_NonNullLiteral(op, value=value, dtype=dtype)


_SIMPLE_OPS = {
    ops.RegexReplace: "_ibis_regex_replace",
    ops.RegexExtract: "_ibis_regex_extract",
    ops.RegexSearch: "_ibis_regex_search",
    ops.Translate: "_ibis_translate",
    ops.Capitalize: "_ibis_capitalize",
    ops.Reverse: "_ibis_reverse",
    ops.RPad: "_ibis_rpad",
    ops.LPad: "_ibis_lpad",
    ops.Repeat: "_ibis_repeat",
    ops.StringAscii: "_ibis_string_ascii",
    ops.ExtractAuthority: "_ibis_extract_authority",
    ops.ExtractFragment: "_ibis_extract_fragment",
    ops.ExtractHost: "_ibis_extract_host",
    ops.ExtractPath: "_ibis_extract_path",
    ops.ExtractProtocol: "_ibis_extract_protocol",
    ops.ExtractUserInfo: "_ibis_extract_user_info",
    ops.BitwiseXor: "_ibis_xor",
    ops.BitwiseNot: "_ibis_inv",
    ops.Modulus: "mod",
    ops.Log10: "log10",
    ops.TypeOf: "typeof",
    ops.BitOr: "_ibis_bit_or",
    ops.BitAnd: "_ibis_bit_and",
    ops.BitXor: "_ibis_bit_xor",
    ops.First: "_ibis_arbitrary_first",
    ops.Last: "_ibis_arbitrary_last",
    ops.Mode: "_ibis_mode",
    ops.Time: "time",
    ops.Date: "date",
}


for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @SQLiteCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @SQLiteCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(SQLiteCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
