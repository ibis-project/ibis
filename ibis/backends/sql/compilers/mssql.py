from __future__ import annotations

import calendar
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import (
    FALSE,
    NULL,
    STAR,
    TRUE,
    AggGen,
    SQLGlotCompiler,
)
from ibis.backends.sql.datatypes import MSSQLType
from ibis.backends.sql.dialects import MSSQL
from ibis.backends.sql.rewrites import (
    FirstValue,
    LastValue,
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
    lower_sample,
    p,
    replace,
    split_select_distinct_with_order_by,
)
from ibis.common.deferred import var

if TYPE_CHECKING:
    from collections.abc import Mapping

    import ibis.expr.operations as ir

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


@replace(p.WindowFunction(p.Reduction & ~p.ReductionVectorizedUDF, order_by=()))
def rewrite_rows_range_order_by_window(_, **kwargs):
    # MSSQL requires an order by in a window frame that has either ROWS or RANGE
    return _.copy(order_by=(_.func.arg,))


class MSSQLCompiler(SQLGlotCompiler):
    __slots__ = ()

    agg = AggGen(supports_order_by=True)

    dialect = MSSQL
    type_mapper = MSSQLType
    rewrites = (
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_rank,
        rewrite_rows_range_order_by_window,
        *SQLGlotCompiler.rewrites,
    )
    post_rewrites = (split_select_distinct_with_order_by,)
    copy_func_args = True

    LOWERED_OPS = {
        ops.Sample: lower_sample(
            supported_methods=("block",), physical_tables_only=True
        ),
    }

    UNSUPPORTED_OPS = (
        ops.ApproxMedian,
        ops.Array,
        ops.ArrayDistinct,
        ops.ArrayFlatten,
        ops.ArrayMap,
        ops.ArraySort,
        ops.ArrayUnion,
        ops.ArgMax,
        ops.ArgMin,
        ops.BitAnd,
        ops.BitOr,
        ops.BitXor,
        ops.Covariance,
        ops.CountDistinctStar,
        ops.DateDiff,
        ops.Kurtosis,
        ops.IntervalAdd,
        ops.IntervalSubtract,
        ops.IntervalMultiply,
        ops.IntervalFloorDivide,
        ops.IsInf,
        ops.IsNan,
        ops.Levenshtein,
        ops.Map,
        ops.Median,
        ops.Mode,
        ops.NthValue,
        ops.RegexExtract,
        ops.RegexReplace,
        ops.RegexSearch,
        ops.RegexSplit,
        ops.RowID,
        ops.StringSplit,
        ops.StringToDate,
        ops.StringToTimestamp,
        ops.StringToTime,
        ops.StructColumn,
        ops.TimestampDiff,
        ops.Unnest,
    )

    SIMPLE_OPS = {
        ops.Atan2: "atn2",
        ops.DateFromYMD: "datefromparts",
        ops.Hash: "checksum",
        ops.Log10: "log10",
        ops.Power: "power",
        ops.Repeat: "replicate",
        ops.Reverse: "reverse",
        ops.StringAscii: "ascii",
        ops.TimestampNow: "sysdatetime",
        ops.Min: "min",
        ops.Max: "max",
        ops.RandomUUID: "newid",
    }

    NAN = sg.func("double", sge.convert("NaN"))
    POS_INF = sg.func("double", sge.convert("Infinity"))
    NEG_INF = sg.func("double", sge.convert("-Infinity"))

    @staticmethod
    def _generate_groups(groups):
        return groups

    @staticmethod
    def _minimize_spec(op, spec):
        if isinstance(func := op.func, ops.Analytic) and not isinstance(
            func, (ops.First, ops.Last, FirstValue, LastValue, ops.NthValue)
        ):
            return None
        return spec

    def to_sqlglot(
        self,
        expr: ir.Expr,
        *,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
    ):
        """Compile an Ibis expression to a sqlglot object."""
        import ibis

        table_expr = expr.as_table()
        conversions = {
            name: ibis.ifelse(table_expr[name], 1, 0).cast(dt.boolean)
            for name, typ in table_expr.schema().items()
            if typ.is_boolean()
        }

        if conversions:
            table_expr = table_expr.mutate(**conversions)
        return super().to_sqlglot(table_expr, limit=limit, params=params)

    def visit_RandomScalar(self, op):
        # By default RAND() will generate the same value for all calls within a
        # query. The standard way to work around this is to pass in a unique
        # value per call, which `CHECKSUM(NEWID())` provides.
        return self.f.rand(self.f.checksum(self.f.newid()))

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
        return sge.paren(self.f.len(self.f.concat("A", arg, "Z")) - 2, copy=False)

    def visit_Substring(self, op, *, arg, start, length):
        start += 1
        start = self.if_(start >= 1, start, start + self.f.length(arg))
        if length is None:
            # We don't need to worry about if start + length is greater than the
            # length of the string, MSSQL will just return the rest of the string
            length = self.f.length(arg)
        return self.f.substring(arg, start, length)

    def visit_GroupConcat(self, op, *, arg, sep, where, order_by):
        if where is not None:
            arg = self.if_(where, arg, NULL)

        out = self.f.group_concat(arg, sep)

        if order_by:
            out = sge.WithinGroup(this=out, expression=sge.Order(expressions=order_by))

        return out

    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.sum(self.if_(where, 1, 0))
        return self.f.count_big(STAR)

    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count_big(sge.Distinct(expressions=[arg]))

    def visit_ApproxQuantile(self, op, *, arg, quantile, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.WithinGroup(
            this=self.f.approx_percentile_cont(quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg, nulls_first=True)]),
        )

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

        return self.f.datetrunc(self.v[unit], arg)

    visit_DateTruncate = visit_TimestampTruncate = visit_DateTimestampTruncate

    def visit_Date(self, op, *, arg):
        return self.cast(arg, dt.date)

    def visit_DateTimeDelta(self, op, *, left, right, part):
        return self.f.datediff(sge.Var(this=part.this.upper()), right, left)

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
            self.f.datediff(self.v.s, "1970-01-01 00:00:00", arg),
            dt.int64,
        )

    def visit_ExtractTemporalComponent(self, op, *, arg):
        return self.f.anon.datepart(
            self.v[type(op).__name__[len("Extract") :].lower()], arg
        )

    visit_ExtractYear = visit_ExtractMonth = visit_ExtractDay = (
        visit_ExtractDayOfYear
    ) = visit_ExtractHour = visit_ExtractMinute = visit_ExtractSecond = (
        visit_ExtractMillisecond
    ) = visit_ExtractMicrosecond = visit_ExtractTemporalComponent

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.anon.datepart(self.v.iso_week, arg)

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
            try:
                this = sge.DataType.Type.UUID
            except AttributeError:
                this = sge.DataType.Type.UNIQUEIDENTIFIER
            return sge.Cast(this=sge.convert(str(value)), to=sge.DataType(this=this))
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
        return self.f.log(arg, 2)

    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.log(arg)
        return self.f.log(arg, base)

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
        elif isinstance(arg, (sge.Window, sge.Max, sge.Min)):
            # special case Window, Max, and Min.
            # These are used for NOT ANY or NOT ALL and friends.
            # We are working around MSSQL's rather unfriendly boolean handling rules
            # and because Max or Min don't return booleans, we have to handle the equality check
            # in a case statement instead.
            # e.g.
            # IFF(MAX(IFF(condition, 1, 0)) = 0, true_case, false_case)
            # is invalid
            # Needs to be
            # CASE WHEN MAX(IFF(condition, 1, 0)) = 0 THEN true_case ELSE false_case END
            return sge.Case(ifs=[self.if_(arg.eq(0), 1)], default=0)
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

    def visit_Any(self, op, *, arg, where):
        arg = self.if_(arg, 1, 0)
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.Max(this=arg)

    def visit_All(self, op, *, arg, where):
        arg = self.if_(arg, 1, 0)
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.Min(this=arg)

    def visit_Select(
        self, op, *, parent, selections, predicates, qualified, sort_keys, distinct
    ):
        # if we've constructed a useless projection return the parent relation
        if not (selections or predicates or qualified or sort_keys or distinct):
            return parent

        result = parent

        if selections:
            result = sg.select(*self._cleanup_names(selections), copy=False).from_(
                result, copy=False
            )

        if predicates:
            result = result.where(*predicates, copy=True)

        if qualified:
            result = result.qualify(*qualified, copy=True)

        if sort_keys:
            result = result.order_by(*sort_keys, copy=False)

        if distinct:
            result = result.distinct()

        return result

    def visit_TimestampAdd(self, op, *, left, right):
        return self.f.dateadd(right.unit, self.cast(right.this, dt.int64), left)

    def visit_TimestampSub(self, op, *, left, right):
        return self.f.dateadd(right.unit, -self.cast(right.this, dt.int64), left)

    visit_DateAdd = visit_TimestampAdd
    visit_DateSub = visit_TimestampSub

    def visit_StartsWith(self, op, *, arg, start):
        return arg.like(self.f.concat(start, "%"))

    def visit_EndsWith(self, op, *, arg, end):
        return arg.like(self.f.concat("%", end))

    def visit_LPad(self, op, *, arg, length, pad):
        return self.if_(
            length <= self.f.length(arg),
            arg,
            self.f.right(
                self.f.concat(self.f.replicate(pad, length - self.f.length(arg)), arg),
                length,
            ),
        )

    def visit_RPad(self, op, *, arg, length, pad):
        return self.if_(
            length <= self.f.length(arg),
            arg,
            self.f.left(
                self.f.concat(arg, self.f.replicate(pad, length - self.f.length(arg))),
                length,
            ),
        )


compiler = MSSQLCompiler()
