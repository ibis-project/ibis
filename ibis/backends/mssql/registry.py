from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)
from ibis.backends.base.sql.alchemy.registry import substr, variance_reduction


def _reduction(func, cast_type="int32"):
    def reduction_compiler(t, op):
        arg, where = op.args

        if arg.dtype.is_boolean():
            if isinstance(arg, ops.TableColumn):
                nullable = arg.dtype.nullable
                arg = ops.Cast(arg, dt.dtype(cast_type)(nullable=nullable))
            else:
                arg = ops.IfElse(arg, 1, 0)

        if where is not None:
            arg = ops.IfElse(where, arg, None)
        return func(t.translate(arg))

    return reduction_compiler


@compiles(substr, "mssql")
def mssql_substr(element, compiler, **kw):
    return compiler.process(sa.func.substring(*element.clauses), **kw)


# String
# TODO: find is copied from SQLite, we should really have a
# "base" set of SQL functions that are the most common APIs across the major
# RDBMS
def _string_find(t, op):
    arg, substr, start, _ = op.args

    sa_arg = t.translate(arg)
    sa_substr = t.translate(substr)

    if start is not None:
        sa_start = t.translate(start)
        return sa.func.charindex(sa_substr, sa_arg, sa_start) - 1

    return sa.func.charindex(sa_substr, sa_arg) - 1


def _extract(fmt):
    def translator(t, op):
        (arg,) = op.args
        sa_arg = t.translate(arg)
        # sa.literal_column is used because it makes the argument pass
        # in NOT as a parameter
        return sa.cast(sa.func.datepart(sa.literal_column(fmt), sa_arg), sa.SMALLINT)

    return translator


def _round(t, op):
    sa_arg = t.translate(op.arg)

    if op.digits is not None:
        return sa.func.round(sa_arg, t.translate(op.digits))
    else:
        return sa.func.round(sa_arg, 0)


def _timestamp_from_unix(x, unit="s"):
    if unit == "s":
        return sa.func.dateadd(sa.text("s"), x, "1970-01-01 00:00:00")
    if unit == "ms":
        return sa.func.dateadd(sa.text("s"), x / 1_000, "1970-01-01 00:00:00")
    raise com.UnsupportedOperationError(f"{unit!r} unit is not supported!")


_interval_units = {
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


def _timestamp_truncate(t, op):
    arg = t.translate(op.arg)
    unit = op.unit.short
    if unit not in _interval_units:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {op.unit!r}")

    return sa.func.datetrunc(sa.text(_interval_units[unit]), arg)


def _timestamp_bucket(t, op):
    unit = op.interval.dtype.unit.short
    if not isinstance(op.interval, ops.Literal):
        raise com.UnsupportedOperationError(
            "Only literal interval values are supported"
        )
    if unit == "us" or unit not in _interval_units:
        raise com.UnsupportedOperationError(
            f"Unsupported bucket interval {op.interval!r}"
        )
    if op.offset is not None:
        raise com.UnsupportedOperationError(
            "Timestamp bucket with offset is not supported"
        )

    part = sa.literal_column(_interval_units[unit])
    value = sa.literal_column(str(op.interval.value))
    arg = t.translate(op.arg)
    origin = sa.literal_column("CAST('1970-01-01' AS DATETIME2)")

    return sa.func.DATE_BUCKET(part, value, arg, origin)


def _temporal_delta(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return sa.func.datediff(sa.literal_column(op.part.value.upper()), right, left)


def _not(t, op):
    arg = t.translate(op.arg)
    if t.within_where:
        return sa.not_(arg)
    else:
        # mssql doesn't support boolean types or comparisons at selection positions
        # so we need to compare the value wrapped in a case statement
        return sa.case((arg == 0, True), else_=False)


operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(sqlalchemy_window_functions_registry)

operation_registry.update(
    {
        ops.Not: _not,
        # aggregate methods
        ops.Count: _reduction(sa.func.count),
        ops.Max: _reduction(sa.func.max),
        ops.Min: _reduction(sa.func.min),
        ops.Sum: _reduction(sa.func.sum),
        ops.Mean: _reduction(sa.func.avg, "float64"),
        ops.IfElse: fixed_arity(sa.func.iif, 3),
        # string methods
        ops.Capitalize: unary(
            lambda arg: sa.func.concat(
                sa.func.upper(sa.func.substring(arg, 1, 1)),
                sa.func.lower(sa.func.substring(arg, 2, sa.func.datalength(arg) - 1)),
            )
        ),
        ops.LStrip: unary(sa.func.ltrim),
        ops.Lowercase: unary(sa.func.lower),
        ops.RStrip: unary(sa.func.rtrim),
        ops.Repeat: fixed_arity(sa.func.replicate, 2),
        ops.Reverse: unary(sa.func.reverse),
        ops.StringFind: _string_find,
        ops.StringLength: unary(sa.func.datalength),
        ops.StringReplace: fixed_arity(sa.func.replace, 3),
        ops.Strip: unary(sa.func.trim),
        ops.Uppercase: unary(sa.func.upper),
        # math
        ops.Abs: unary(sa.func.abs),
        ops.Acos: unary(sa.func.acos),
        ops.Asin: unary(sa.func.asin),
        ops.Atan2: fixed_arity(sa.func.atn2, 2),
        ops.Atan: unary(sa.func.atan),
        ops.Ceil: unary(sa.func.ceiling),
        ops.Cos: unary(sa.func.cos),
        ops.Floor: unary(sa.func.floor),
        ops.FloorDivide: fixed_arity(
            lambda left, right: sa.func.floor(left / right), 2
        ),
        ops.Power: fixed_arity(sa.func.power, 2),
        ops.Sign: unary(sa.func.sign),
        ops.Sin: unary(sa.func.sin),
        ops.Sqrt: unary(sa.func.sqrt),
        ops.Tan: unary(sa.func.tan),
        ops.Round: _round,
        ops.RandomScalar: fixed_arity(sa.func.RAND, 0),
        ops.Ln: fixed_arity(sa.func.log, 1),
        ops.Log: fixed_arity(lambda x, p: sa.func.log(x, p), 2),
        ops.Log2: fixed_arity(lambda x: sa.func.log(x, 2), 1),
        ops.Log10: fixed_arity(lambda x: sa.func.log(x, 10), 1),
        ops.StandardDev: variance_reduction("stdev", {"sample": "", "pop": "p"}),
        ops.Variance: variance_reduction("var", {"sample": "", "pop": "p"}),
        # timestamp methods
        ops.TimestampNow: fixed_arity(sa.func.GETDATE, 0),
        ops.ExtractYear: _extract("year"),
        ops.ExtractMonth: _extract("month"),
        ops.ExtractDay: _extract("day"),
        ops.ExtractDayOfYear: _extract("dayofyear"),
        ops.ExtractHour: _extract("hour"),
        ops.ExtractMinute: _extract("minute"),
        ops.ExtractSecond: _extract("second"),
        ops.ExtractMillisecond: _extract("millisecond"),
        ops.ExtractWeekOfYear: _extract("iso_week"),
        ops.DayOfWeekIndex: fixed_arity(
            lambda x: sa.func.datepart(sa.text("weekday"), x) - 1, 1
        ),
        ops.ExtractEpochSeconds: fixed_arity(
            lambda x: sa.cast(
                sa.func.datediff(sa.text("s"), "1970-01-01 00:00:00", x), sa.BIGINT
            ),
            1,
        ),
        ops.TimestampFromUNIX: lambda t, op: _timestamp_from_unix(
            t.translate(op.arg), op.unit.short
        ),
        ops.DateFromYMD: fixed_arity(sa.func.datefromparts, 3),
        ops.TimestampFromYMDHMS: fixed_arity(
            lambda y, m, d, h, min, s: sa.func.datetimefromparts(y, m, d, h, min, s, 0),
            6,
        ),
        ops.TimeFromHMS: fixed_arity(
            lambda h, m, s: sa.func.timefromparts(h, m, s, 0, 0), 3
        ),
        ops.TimestampTruncate: _timestamp_truncate,
        ops.DateTruncate: _timestamp_truncate,
        ops.TimestampBucket: _timestamp_bucket,
        ops.Hash: unary(sa.func.checksum),
        ops.ExtractMicrosecond: fixed_arity(
            lambda arg: sa.func.datepart(sa.literal_column("microsecond"), arg), 1
        ),
        ops.TimeDelta: _temporal_delta,
        ops.DateDelta: _temporal_delta,
        ops.TimestampDelta: _temporal_delta,
    }
)

_invalid_operations = {
    # ibis.expr.operations.strings
    ops.RPad,
    ops.LPad,
    # ibis.expr.operations.reductions
    ops.BitAnd,
    ops.BitOr,
    ops.BitXor,
    ops.GroupConcat,
    # ibis.expr.operations.window
    ops.NthValue,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
