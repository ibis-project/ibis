from __future__ import annotations

import calendar

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import (
    FALSE,
    NULL,
    STAR,
    TRUE,
    SQLGlotCompiler,
    paren,
)
from ibis.backends.base.sqlglot.datatypes import MSSQLType
from ibis.backends.base.sqlglot.dialects import MSSQL
from ibis.backends.base.sqlglot.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_row_number,
    rewrite_first_to_first_value,
    rewrite_last_to_last_value,
    rewrite_sample_as_filter,
)
from ibis.common.deferred import var

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


@public
class MSSQLCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = MSSQL
    type_mapper = MSSQLType
    rewrites = (
        rewrite_sample_as_filter,
        rewrite_first_to_first_value,
        rewrite_last_to_last_value,
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_row_number,
        *SQLGlotCompiler.rewrites,
    )

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.Any,
            ops.All,
            ops.ApproxMedian,
            ops.Arbitrary,
            ops.ArgMax,
            ops.ArgMin,
            ops.ArrayCollect,
            ops.Array,
            ops.ArrayDistinct,
            ops.ArrayFlatten,
            ops.ArrayMap,
            ops.ArraySort,
            ops.ArrayUnion,
            ops.BitAnd,
            ops.BitOr,
            ops.BitXor,
            ops.Covariance,
            ops.CountDistinctStar,
            ops.DateAdd,
            ops.DateDiff,
            ops.DateSub,
            ops.EndsWith,
            ops.First,
            ops.IntervalAdd,
            ops.IntervalFromInteger,
            ops.IntervalMultiply,
            ops.IntervalSubtract,
            ops.IsInf,
            ops.IsNan,
            ops.Last,
            ops.LPad,
            ops.Levenshtein,
            ops.Map,
            ops.Median,
            ops.Mode,
            ops.MultiQuantile,
            ops.NthValue,
            ops.Quantile,
            ops.RegexExtract,
            ops.RegexReplace,
            ops.RegexSearch,
            ops.RegexSplit,
            ops.RowID,
            ops.RPad,
            ops.StartsWith,
            ops.StringSplit,
            ops.StringToTimestamp,
            ops.StructColumn,
            ops.TimestampAdd,
            ops.TimestampDiff,
            ops.TimestampSub,
            ops.Unnest,
        )
    )

    SIMPLE_OPS = {
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

    def visit_GroupConcat(self, op, *, arg, sep, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.group_concat(arg, sep)

    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.sum(self.if_(where, 1, 0))
        return self.f.count(STAR)

    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.f.datepart(self.v.weekday, arg) - 1

    def visit_DayOfWeekName(self, op, *, arg):
        days = calendar.day_name
        return sge.Case(
            this=self.f.datepart(self.v.weekday, arg) - 1,
            ifs=list(map(self.if_, *zip(*enumerate(days)))),
        )

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

    visit_DateTruncate = visit_TimestampTruncate = visit_DateTimestampTruncate

    def visit_Date(self, op, *, arg):
        return self.cast(arg, dt.date)

    def visit_DateTimeDelta(self, op, *, left, right, part):
        return self.f.datediff(
            sge.Var(this=part.this.upper()), right, left, dialect=self.dialect
        )

    visit_TimeDelta = visit_DateDelta = visit_TimestampDelta = visit_DateTimeDelta

    def visit_Xor(self, op, *, left, right):
        return sg.and_(sg.or_(left, right), sg.not_(sg.and_(left, right)))

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

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.cast(
            self.f.datediff(self.v.s, "1970-01-01 00:00:00", arg, dialect=self.dialect),
            dt.int64,
        )

    def visit_ExtractTemporalComponent(self, op, *, arg):
        return self.f.datepart(self.v[type(op).__name__[len("Extract") :].lower()], arg)

    visit_ExtractYear = (
        visit_ExtractMonth
    ) = (
        visit_ExtractDay
    ) = (
        visit_ExtractDayOfYear
    ) = (
        visit_ExtractHour
    ) = (
        visit_ExtractMinute
    ) = (
        visit_ExtractSecond
    ) = (
        visit_ExtractMillisecond
    ) = visit_ExtractMicrosecond = visit_ExtractTemporalComponent

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.datepart(self.v.iso_week, arg)

    def visit_TimeFromHMS(self, op, *, hours, minutes, seconds):
        return self.f.timefromparts(hours, minutes, seconds, 0, 0)

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        return self.f.datetimefromparts(year, month, day, hours, minutes, seconds, 0)

    def visit_StringFind(self, op, *, arg, substr, start, end):
        if start is not None:
            return self.f.charindex(substr, arg, start)
        return self.f.charindex(substr, arg)

    def visit_Round(self, op, *, arg, digits):
        return self.f.round(arg, digits if digits is not None else 0)

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

    def visit_Log2(self, op, *, arg):
        return self.f.log(arg, 2, dialect=self.dialect)

    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.log(arg, dialect=self.dialect)
        return self.f.log(arg, base, dialect=self.dialect)

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype

        if to.is_boolean():
            # no such thing as a boolean in MSSQL
            return arg
        elif from_.is_integer() and to.is_timestamp():
            return self.f.dateadd(self.v.s, arg, "1970-01-01 00:00:00")
        return super().visit_Cast(op, arg=arg, to=to)

    def visit_Sum(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.if_(arg, 1, 0)
        return self.agg.sum(arg, where=where)

    def visit_Mean(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.if_(arg, 1, 0)
        return self.agg.avg(arg, where=where)

    def visit_Not(self, op, *, arg):
        if isinstance(arg, sge.Boolean):
            return FALSE if arg == TRUE else TRUE
        return self.if_(arg, 1, 0).eq(0)

    def visit_HashBytes(self, op, *, arg, how):
        if how in ("md5", "sha1"):
            return self.f.hashbytes(how, arg)
        elif how == "sha256":
            return self.f.hashbytes("sha2_256", arg)
        elif how == "sha512":
            return self.f.hashbytes("sha2_512", arg)
        else:
            raise NotImplementedError(how)

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

    def visit_StringConcat(self, op, *, arg):
        any_args_null = (a.is_(NULL) for a in arg)
        return self.if_(sg.or_(*any_args_null), NULL, self.f.concat(*arg))
