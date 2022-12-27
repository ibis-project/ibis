import sqlalchemy as sa

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)


def _reduction(func, cast_type='int32'):
    def reduction_compiler(t, op):
        arg, where = op.args

        if arg.output_dtype.is_boolean():
            nullable = arg.output_dtype.nullable
            arg = ops.Cast(arg, dt.dtype(cast_type)(nullable=nullable))

        if where is not None:
            arg = ops.Where(where, arg, None)
        return func(t.translate(arg))

    return reduction_compiler


# String
# TODO: substr and find are copied from SQLite, we should really have a
# "base" set of SQL functions that are the most common APIs across the major
# RDBMS
def _substr(t, op):
    f = sa.func.substring

    arg, start, length = op.args

    sa_arg = t.translate(arg)
    sa_start = t.translate(start)

    if length is None:
        return f(sa_arg, sa_start + 1)
    else:
        sa_length = t.translate(length)
        return f(sa_arg, sa_start + 1, sa_length)


def _string_find(t, op):
    arg, substr, start, _ = op.args

    sa_arg = t.translate(arg)
    sa_substr = t.translate(substr)

    if start is not None:
        sa_start = t.translate(start)
        return sa.func.charindex(sa_substr, sa_arg, sa_start) - 1

    return sa.func.charindex(sa_substr, sa_arg) - 1


# Numerical
def _floor_divide(t, op):
    left, right = map(t.translate, op.args)
    return sa.func.floor(left / right)


def _extract(fmt):
    def translator(t, op):
        (arg,) = op.args
        sa_arg = t.translate(arg)
        # sa.literal_column is used becuase it makes the argument pass
        # in NOT as a parameter
        return sa.cast(sa.func.datepart(sa.literal_column(fmt), sa_arg), sa.SMALLINT)

    return translator


def _round(t, op):
    sa_arg = t.translate(op.arg)

    if op.digits is not None:
        return sa.func.round(sa_arg, t.translate(op.digits))
    else:
        return sa.func.round(sa_arg, 0)


operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(sqlalchemy_window_functions_registry)

operation_registry.update(
    {
        # aggregate methods
        ops.Count: _reduction(sa.func.count),
        ops.Max: _reduction(sa.func.max),
        ops.Min: _reduction(sa.func.min),
        ops.Sum: _reduction(sa.func.sum),
        ops.Mean: _reduction(sa.func.avg, 'float64'),
        ops.Where: fixed_arity(sa.func.iif, 3),
        # string methods
        ops.LStrip: unary(sa.func.ltrim),
        ops.Lowercase: unary(sa.func.lower),
        ops.RStrip: unary(sa.func.rtrim),
        ops.Repeat: fixed_arity(sa.func.replicate, 2),
        ops.Reverse: unary(sa.func.reverse),
        ops.StringFind: _string_find,
        ops.StringLength: unary(sa.func.datalength),
        ops.StringReplace: fixed_arity(sa.func.replace, 3),
        ops.Strip: unary(sa.func.trim),
        ops.Substring: _substr,
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
        ops.FloorDivide: _floor_divide,
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
        # timestamp methods
        ops.TimestampNow: fixed_arity(sa.func.GETDATE, 0),
        ops.ExtractYear: _extract('year'),
        ops.ExtractMonth: _extract('month'),
        ops.ExtractDay: _extract('day'),
        ops.ExtractDayOfYear: _extract('dayofyear'),
        ops.ExtractHour: _extract('hour'),
        ops.ExtractMinute: _extract('minute'),
        ops.ExtractSecond: _extract('second'),
        ops.ExtractMillisecond: _extract('millisecond'),
        ops.ExtractWeekOfYear: _extract('iso_week'),
    }
)
