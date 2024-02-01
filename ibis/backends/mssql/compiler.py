from __future__ import annotations

import calendar
from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public
from sqlglot.dialects import TSQL
from sqlglot.dialects.dialect import rename_func

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import NULL, STAR, SQLGlotCompiler, paren
from ibis.backends.base.sqlglot.datatypes import MSSQLType
from ibis.backends.base.sqlglot.rewrites import (
    rewrite_first_to_first_value,
    rewrite_last_to_last_value,
)
from ibis.common.deferred import var
from ibis.common.patterns import replace
from ibis.expr.rewrites import p, rewrite_sample

TSQL.Generator.TRANSFORMS |= {
    sge.ApproxDistinct: rename_func("approx_count_distinct"),
    sge.Stddev: rename_func("stdevp"),
    sge.StddevPop: rename_func("stdevp"),
    sge.StddevSamp: rename_func("stdev"),
    sge.Variance: rename_func("var"),
    sge.VariancePop: rename_func("varp"),
    sge.Ceil: rename_func("ceiling"),
    sge.Trim: lambda self, e: f"TRIM({e.this.sql(self.dialect)})",
    sge.DateFromParts: rename_func("datefromparts"),
}

y = var("y")
start = var("start")
end = var("end")


# MS SQL facts that make using it a nightmare:
#
# * There is no boolean type
# * There are no boolean literals
# * But there's a numeric bit type whose domain is THE TWO VALUES 0 and 1 (and NULL of course), seriously?
# * Supported boolean expressions are =, <>, <, >, <=, >=, IS NULL,
#   IS NOT NULL, IN, NOT IN, EXISTS, BETWEEN, IS NOT DISTINCT FROM, IS DISTINCT FROM,
#   LIKE, NOT LIKE, CONTAINS (?), ALL, SOME, ANY
#   The return type of these is anyone's guess, but it's definitely NOT BOOLEAN
# * Boolean expressions CANNOT be used in a projection, i.e., SELECT x = 1 is not allowed
# * Boolean expressions MUST be used in a WHERE clause, i.e., SELECT * FROM t WHERE 1 is not allowed


@replace(p.WindowFunction(p.RowNumber | p.NTile, y))
def exclude_unsupported_window_frame_from_ops_with_offset(_, y):
    return ops.Subtract(_.copy(frame=y.copy(start=None, end=0)), 1)


@replace(p.WindowFunction(p.Lag | p.Lead | p.PercentRank | p.CumeDist, y))
def exclude_unsupported_window_frame_from_ops(_, y):
    return _.copy(frame=y.copy(start=None, end=0))


@public
class MSSQLCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "tsql"
    type_mapper = MSSQLType
    rewrites = (
        rewrite_sample,
        rewrite_first_to_first_value,
        rewrite_last_to_last_value,
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_ops_with_offset,
        *SQLGlotCompiler.rewrites,
    )
    quoted = True

    @property
    def NAN(self):
        return self.f.double("NaN")

    @property
    def POS_INF(self):
        return self.f.double("Infinity")

    @property
    def NEG_INF(self):
        return self.f.double("-Infinity")

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]
        if where is not None:
            args = tuple(self.if_(where, arg, NULL) for arg in args)
        return func(*args)

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @staticmethod
    def _generate_groups(groups):
        return groups

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

    @visit_node.register(ops.StringLength)
    def visit_StringLength(self, op, *, arg):
        """The MSSQL LEN function doesn't count trailing spaces.

        Also, DATALENGTH (the suggested alternative) counts bytes and thus its
        result depends on the string's encoding.

        https://learn.microsoft.com/en-us/sql/t-sql/functions/len-transact-sql?view=sql-server-ver16#remarks

        The solution is to add a character to the beginning and end of the
        string that are guaranteed to have one character in length and are not
        spaces, and then subtract 2 from the result of `LEN` of that input.

        Thanks to @arkanovicz for this glorious hack.
        """
        return paren(self.f.len(self.f.concat("A", arg, "Z")) - 2)

    @visit_node.register(ops.Capitalize)
    def visit_Capitalize(self, op, *, arg):
        length = paren(self.f.len(self.f.concat("A", arg, "Z")) - 2)
        return self.f.concat(
            self.f.upper(self.f.substring(arg, 1, 1)),
            self.f.lower(self.f.substring(arg, 2, length - 1)),
        )

    @visit_node.register(ops.GroupConcat)
    def visit_GroupConcat(self, op, *, arg, sep, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.group_concat(arg, sep)

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.sum(self.if_(where, 1, 0))
        return self.f.count(STAR)

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.f.datepart(self.v.weekday, arg) - 1

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg):
        days = calendar.day_name
        return sge.Case(
            this=self.f.datepart(self.v.weekday, arg) - 1,
            ifs=list(map(self.if_, *zip(*enumerate(days)))),
        )

    @visit_node.register(ops.DateTruncate)
    @visit_node.register(ops.TimestampTruncate)
    def visit_DateTimestampTruncate(self, op, *, arg, unit):
        interval_units = {
            "us": "microsecond",
            "ms": "millisecond",
            "s": "second",
            "m": "minute",
            "h": "hour",
            "D": "day",
            "W": "week",
            "M": "month",
            "Q": "quarter",
            "Y": "year",
        }
        if (unit := interval_units.get(unit.short)) is None:
            raise com.UnsupportedOperationError(f"Unsupported truncate unit {unit!r}")

        return self.f.datetrunc(self.v[unit], arg, dialect=self.dialect)

    @visit_node.register(ops.Date)
    def visit_Date(self, op, *, arg):
        return self.cast(arg, dt.date)

    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.TimestampDelta)
    def visit_DateTimeDelta(self, op, *, left, right, part):
        return self.f.datediff(
            sge.Var(this=part.this.upper()), right, left, dialect=self.dialect
        )

    @visit_node.register(ops.Xor)
    def visit_Xor(self, op, *, left, right):
        return sg.and_(sg.or_(left, right), sg.not_(sg.and_(left, right)))

    @visit_node.register(ops.TimestampBucket)
    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        interval_units = {
            "ms": "millisecond",
            "s": "second",
            "m": "minute",
            "h": "hour",
            "D": "day",
            "W": "week",
            "M": "month",
            "Q": "quarter",
            "Y": "year",
        }

        if not isinstance(op.interval, ops.Literal):
            raise com.UnsupportedOperationError(
                "Only literal interval values are supported with MS SQL timestamp bucketing"
            )

        if (unit := interval_units.get(op.interval.dtype.unit.short)) is None:
            raise com.UnsupportedOperationError(
                f"Unsupported bucket interval {op.interval!r}"
            )
        if offset is not None:
            raise com.UnsupportedOperationError(
                "Timestamp bucket with offset is not supported"
            )

        part = self.v[unit]
        origin = self.cast("1970-01-01", op.arg.dtype)

        return self.f.date_bucket(part, op.interval.value, arg, origin)

    @visit_node.register(ops.ExtractEpochSeconds)
    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.cast(
            self.f.datediff(self.v.s, "1970-01-01 00:00:00", arg, dialect=self.dialect),
            dt.int64,
        )

    @visit_node.register(ops.ExtractYear)
    @visit_node.register(ops.ExtractMonth)
    @visit_node.register(ops.ExtractDay)
    @visit_node.register(ops.ExtractDayOfYear)
    @visit_node.register(ops.ExtractHour)
    @visit_node.register(ops.ExtractMinute)
    @visit_node.register(ops.ExtractSecond)
    @visit_node.register(ops.ExtractMillisecond)
    @visit_node.register(ops.ExtractMicrosecond)
    def visit_Extract(self, op, *, arg):
        return self.f.datepart(self.v[type(op).__name__[len("Extract") :].lower()], arg)

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.datepart(self.v.iso_week, arg)

    @visit_node.register(ops.TimeFromHMS)
    def visit_TimeFromHMS(self, op, *, hours, minutes, seconds):
        return self.f.timefromparts(hours, minutes, seconds, 0, 0)

    @visit_node.register(ops.TimestampFromYMDHMS)
    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        return self.f.datetimefromparts(year, month, day, hours, minutes, seconds, 0)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if start is not None:
            return self.f.charindex(substr, arg, start)
        return self.f.charindex(substr, arg)

    @visit_node.register(ops.Round)
    def visit_Round(self, op, *, arg, digits):
        return self.f.round(arg, digits if digits is not None else 0)

    @visit_node.register(ops.TimestampFromUNIX)
    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        unit = unit.short
        if unit == "s":
            return self.f.dateadd(self.v.s, arg, "1970-01-01 00:00:00")
        elif unit == "ms":
            return self.f.dateadd(self.v.s, arg / 1_000, "1970-01-01 00:00:00")
        raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_decimal():
            return self.cast(str(value.normalize()), dtype)
        elif dtype.is_date():
            return self.f.datefromparts(value.year, value.month, value.day)
        elif dtype.is_timestamp():
            args = (
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
                value.microsecond,
            )
            if dtype.timezone is not None:
                assert value.tzinfo is not None

                offset = value.strftime("%z")
                hour_offset = int(offset[:3])
                minute_offset = int(offset[-2:])
                return self.f.datetimeoffsetfromparts(
                    *args, hour_offset, minute_offset, 6
                )
            else:
                return self.f.datetime2fromparts(*args, 6)
        elif dtype.is_time():
            return self.f.timefromparts(
                value.hour, value.minute, value.second, value.microsecond, 0
            )
        elif dtype.is_uuid():
            return sge.Cast(
                this=sge.convert(str(value)),
                to=sge.DataType(this=sge.DataType.Type.UNIQUEIDENTIFIER),
            )
        elif dtype.is_binary():
            return self.f.convert(
                sge.DataType(this=sge.DataType.Type.VARBINARY),
                value.hex(),
                2,  # style, see https://learn.microsoft.com/en-us/sql/t-sql/functions/cast-and-convert-transact-sql?view=sql-server-ver16#binary-styles
            )
        elif dtype.is_array() or dtype.is_struct() or dtype.is_map():
            raise com.UnsupportedBackendType("MS SQL does not support complex types")

        return None

    @visit_node.register(ops.Log2)
    def visit_Log2(self, op, *, arg):
        return self.f.log(arg, 2, dialect=self.dialect)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.log(arg, dialect=self.dialect)
        return self.f.log(arg, base, dialect=self.dialect)

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype

        if to.is_boolean():
            # no such thing as a boolean in MSSQL
            return arg
        elif from_.is_integer() and to.is_timestamp():
            return self.f.dateadd(self.v.s, arg, "1970-01-01 00:00:00")
        return super().visit_Cast(op, arg=arg, to=to)

    @visit_node.register(ops.Sum)
    def visit_Sum(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.if_(arg, 1, 0)
        return self.agg.sum(arg, where=where)

    @visit_node.register(ops.Mean)
    def visit_Mean(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.if_(arg, 1, 0)
        return self.agg.avg(arg, where=where)

    @visit_node.register(ops.Not)
    def visit_Not(self, op, *, arg):
        if isinstance(arg, sge.Boolean):
            return sge.FALSE if arg == sge.TRUE else sge.TRUE
        return self.if_(arg, 1, 0).eq(0)

    @visit_node.register(ops.HashBytes)
    def visit_HashBytes(self, op, *, arg, how):
        if how in ("md5", "sha1"):
            return self.f.hashbytes(how, arg)
        elif how == "sha256":
            return self.f.hashbytes("sha2_256", arg)
        elif how == "sha512":
            return self.f.hashbytes("sha2_512", arg)
        else:
            raise NotImplementedError(how)

    @visit_node.register(ops.HexDigest)
    def visit_HexDigest(self, op, *, arg, how):
        if how in ("md5", "sha1"):
            hashbinary = self.f.hashbytes(how, arg)
        elif how == "sha256":
            hashbinary = self.f.hashbytes("sha2_256", arg)
        elif how == "sha512":
            hashbinary = self.f.hashbytes("sha2_512", arg)
        else:
            raise NotImplementedError(how)

        # mssql uppercases the hexdigest which is inconsistent with several other
        # implementations and inconsistent with Python, so lowercase it.
        return self.f.lower(
            self.f.convert(
                sge.Literal(this="VARCHAR(MAX)", is_string=False), hashbinary, 2
            )
        )

    @visit_node.register(ops.Any)
    @visit_node.register(ops.All)
    @visit_node.register(ops.ApproxMedian)
    @visit_node.register(ops.Arbitrary)
    @visit_node.register(ops.ArgMax)
    @visit_node.register(ops.ArgMin)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.Array)
    @visit_node.register(ops.ArrayDistinct)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.ArraySort)
    @visit_node.register(ops.ArrayUnion)
    @visit_node.register(ops.BitAnd)
    @visit_node.register(ops.BitOr)
    @visit_node.register(ops.BitXor)
    @visit_node.register(ops.Covariance)
    @visit_node.register(ops.CountDistinctStar)
    @visit_node.register(ops.DateAdd)
    @visit_node.register(ops.DateDiff)
    @visit_node.register(ops.DateSub)
    @visit_node.register(ops.EndsWith)
    @visit_node.register(ops.First)
    @visit_node.register(ops.IntervalAdd)
    @visit_node.register(ops.IntervalFromInteger)
    @visit_node.register(ops.IntervalMultiply)
    @visit_node.register(ops.IntervalSubtract)
    @visit_node.register(ops.IsInf)
    @visit_node.register(ops.IsNan)
    @visit_node.register(ops.Last)
    @visit_node.register(ops.LPad)
    @visit_node.register(ops.Levenshtein)
    @visit_node.register(ops.Map)
    @visit_node.register(ops.Median)
    @visit_node.register(ops.Mode)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.NthValue)
    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.RegexExtract)
    @visit_node.register(ops.RegexReplace)
    @visit_node.register(ops.RegexSearch)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.RowID)
    @visit_node.register(ops.RPad)
    @visit_node.register(ops.StartsWith)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.StringToTimestamp)
    @visit_node.register(ops.StructColumn)
    @visit_node.register(ops.TimestampAdd)
    @visit_node.register(ops.TimestampDiff)
    @visit_node.register(ops.TimestampSub)
    @visit_node.register(ops.Unnest)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
    ops.Atan2: "atn2",
    ops.DateFromYMD: "datefromparts",
    ops.Hash: "checksum",
    ops.Ln: "log",
    ops.Log10: "log10",
    ops.Power: "power",
    ops.RandomScalar: "rand",
    ops.Repeat: "replicate",
    ops.Reverse: "reverse",
    ops.StringAscii: "ascii",
    ops.TimestampNow: "sysdatetime",
    ops.Min: "min",
    ops.Max: "max",
}


for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @MSSQLCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @MSSQLCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(MSSQLCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
