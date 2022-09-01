import functools
import operator

import sqlalchemy as sa
import toolz
from multipledispatch import Dispatcher

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql.alchemy import (
    AlchemyExprTranslator,
    fixed_arity,
    reduction,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
    varargs,
    variance_reduction,
)
from ibis.backends.base.sql.alchemy.registry import _clip, _gen_string_find

operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(sqlalchemy_window_functions_registry)


sqlite_cast = Dispatcher("sqlite_cast")


@sqlite_cast.register(AlchemyExprTranslator, ir.IntegerValue, dt.Timestamp)
def _unixepoch(t, arg, _):
    return sa.func.datetime(t.translate(arg), "unixepoch")


@sqlite_cast.register(AlchemyExprTranslator, ir.StringValue, dt.Timestamp)
def _string_to_timestamp(t, arg, _):
    return sa.func.strftime('%Y-%m-%d %H:%M:%f', t.translate(arg))


@sqlite_cast.register(AlchemyExprTranslator, ir.IntegerValue, dt.Date)
def _integer_to_date(t, arg, _):
    return sa.func.date(sa.func.datetime(t.translate(arg), "unixepoch"))


@sqlite_cast.register(
    AlchemyExprTranslator,
    (ir.StringValue, ir.TimestampValue),
    dt.Date,
)
def _string_or_timestamp_to_date(t, arg, _):
    return sa.func.date(t.translate(arg))


@sqlite_cast.register(
    AlchemyExprTranslator,
    ir.Value,
    (dt.Date, dt.Timestamp),
)
def _value_to_temporal(t, arg, _):
    raise com.UnsupportedOperationError(type(arg))


@sqlite_cast.register(AlchemyExprTranslator, ir.CategoryValue, dt.Int32)
def _category_to_int(t, arg, _):
    return t.translate(arg)


@sqlite_cast.register(AlchemyExprTranslator, ir.Value, dt.DataType)
def _default_cast_impl(t, arg, target_type):
    return sa.cast(t.translate(arg), t.get_sqla_type(target_type))


def _cast(t, expr):
    op = expr.op()
    return sqlite_cast(t, op.arg, op.to)


def _string_right(t, expr):
    f = sa.func.substr

    arg, length = expr.op().args

    sa_arg = t.translate(arg)
    sa_length = t.translate(length)

    return f(sa_arg, -sa_length, sa_length)


def _strftime(t, expr):
    arg, format = expr.op().args
    sa_arg = t.translate(arg)
    sa_format = t.translate(format)
    return sa.func.strftime(sa_format, sa_arg)


def _strftime_int(fmt):
    def translator(t, expr):
        return t.translate(expr.op().arg.strftime(fmt).cast(dt.int32))

    return translator


def _extract_quarter(t, expr):
    (arg,) = expr.op().args

    expr_new = ops.ExtractMonth(arg).to_expr()
    expr_new = (
        ibis.case()
        .when(expr_new.isin([1, 2, 3]), 1)
        .when(expr_new.isin([4, 5, 6]), 2)
        .when(expr_new.isin([7, 8, 9]), 3)
        .else_(4)
        .end()
    )
    return sa.cast(t.translate(expr_new), sa.Integer)


def _extract_epoch_seconds(t, expr):
    (arg,) = expr.op().args
    # example: (julianday('now') - 2440587.5) * 86400.0
    sa_expr = (sa.func.julianday(t.translate(arg)) - 2440587.5) * 86400.0
    return sa.cast(sa_expr, sa.BigInteger)


_truncate_modifiers = {
    'Y': 'start of year',
    'M': 'start of month',
    'D': 'start of day',
    'W': 'weekday 1',
}


def _truncate(func):
    def translator(t, expr):
        arg, unit = expr.op().args
        sa_arg = t.translate(arg)
        try:
            modifier = _truncate_modifiers[unit]
        except KeyError:
            raise com.UnsupportedOperationError(
                f'Unsupported truncate unit {unit!r}'
            )
        return func(sa_arg, modifier)

    return translator


def _millisecond(t, expr):
    (arg,) = expr.op().args
    sa_arg = t.translate(arg)
    fractional_second = sa.func.strftime('%f', sa_arg)
    return (fractional_second * 1000) % 1000


def _log(t, expr):
    arg, base = expr.op().args
    sa_arg = t.translate(arg)
    if base is None:
        return sa.func._ibis_sqlite_ln(sa_arg)
    return sa.func._ibis_sqlite_log(sa_arg, t.translate(base))


def _repeat(t, expr):
    arg, times = map(t.translate, expr.op().args)
    f = sa.func
    return f.replace(
        f.substr(f.quote(f.zeroblob((times + 1) / 2)), 3, times), '0', arg
    )


def _generic_pad(arg, length, pad):
    f = sa.func
    arg_length = f.length(arg)
    pad_length = f.length(pad)
    number_of_zero_bytes = (
        (length - arg_length - 1 + pad_length) / pad_length + 1
    ) / 2
    return f.substr(
        f.replace(
            f.replace(
                f.substr(f.quote(f.zeroblob(number_of_zero_bytes)), 3), "'", ''
            ),
            '0',
            pad,
        ),
        1,
        length - f.length(arg),
    )


def _lpad(t, expr):
    arg, length, pad = map(t.translate, expr.op().args)
    return _generic_pad(arg, length, pad) + arg


def _rpad(t, expr):
    arg, length, pad = map(t.translate, expr.op().args)
    return arg + _generic_pad(arg, length, pad)


def _extract_week_of_year(t, expr):
    """ISO week of year.

    This solution is based on https://stackoverflow.com/a/15511864 and handle
    the edge cases when computing ISO week from non-ISO week.

    The implementation gives the same results as `datetime.isocalendar()`.

    The year's week that "wins" the day is the year with more alloted days.

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
    days in the week occuring in the _previous_ week's year.

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
    days in the week occuring in the _next_ week's year.
    """
    arg = t.translate(expr.op().arg)
    return (
        sa.func.strftime("%j", sa.func.date(arg, "-3 days", "weekday 4")) - 1
    ) / 7 + 1


def _string_join(t, expr):
    sep, elements = expr.op().args
    return functools.reduce(
        operator.add,
        toolz.interpose(t.translate(sep), map(t.translate, elements)),
    )


def _string_concat(t, expr):
    # yes, `arg`. for variadic functions `arg` is the list of arguments.
    #
    # `args` is always the list of values of the fields declared in the
    # operation
    args = expr.op().arg
    return functools.reduce(operator.add, map(t.translate, args))


def _date_from_ymd(t, expr):
    y, m, d = map(t.translate, expr.op().args)
    ymdstr = sa.func.printf('%04d-%02d-%02d', y, m, d)
    return sa.func.date(ymdstr)


def _timestamp_from_ymdhms(t, expr):
    y, mo, d, h, m, s, *rest = (
        t.translate(x) if x is not None else None for x in expr.op().args
    )
    tz = rest[0] if rest else ''
    timestr = sa.func.printf(
        '%04d-%02d-%02d %02d:%02d:%02d%s', y, mo, d, h, m, s, tz
    )
    return sa.func.datetime(timestr)


def _time_from_hms(t, expr):
    h, m, s = map(t.translate, expr.op().args)
    timestr = sa.func.printf('%02d:%02d:%02d', h, m, s)
    return sa.func.time(timestr)


operation_registry.update(
    {
        ops.Cast: _cast,
        ops.DateFromYMD: _date_from_ymd,
        ops.StrRight: _string_right,
        ops.StringFind: _gen_string_find(sa.func.instr),
        ops.StringJoin: _string_join,
        ops.StringConcat: _string_concat,
        ops.Least: varargs(sa.func.min),
        ops.Greatest: varargs(sa.func.max),
        ops.IfNull: fixed_arity(sa.func.ifnull, 2),
        ops.DateFromYMD: _date_from_ymd,
        ops.TimeFromHMS: _time_from_hms,
        ops.TimestampFromYMDHMS: _timestamp_from_ymdhms,
        ops.DateTruncate: _truncate(sa.func.date),
        ops.Date: unary(sa.func.date),
        ops.TimestampTruncate: _truncate(sa.func.datetime),
        ops.Strftime: _strftime,
        ops.ExtractYear: _strftime_int('%Y'),
        ops.ExtractMonth: _strftime_int('%m'),
        ops.ExtractDay: _strftime_int('%d'),
        ops.ExtractWeekOfYear: _extract_week_of_year,
        ops.ExtractDayOfYear: _strftime_int('%j'),
        ops.ExtractQuarter: _extract_quarter,
        ops.ExtractEpochSeconds: _extract_epoch_seconds,
        ops.ExtractHour: _strftime_int('%H'),
        ops.ExtractMinute: _strftime_int('%M'),
        ops.ExtractSecond: _strftime_int('%S'),
        ops.ExtractMillisecond: _millisecond,
        ops.TimestampNow: fixed_arity(lambda: sa.func.datetime("now"), 0),
        ops.RegexSearch: fixed_arity(sa.func._ibis_sqlite_regex_search, 2),
        ops.RegexReplace: fixed_arity(sa.func._ibis_sqlite_regex_replace, 3),
        ops.RegexExtract: fixed_arity(sa.func._ibis_sqlite_regex_extract, 3),
        ops.LPad: _lpad,
        ops.RPad: _rpad,
        ops.Repeat: _repeat,
        ops.Reverse: unary(sa.func._ibis_sqlite_reverse),
        ops.StringAscii: unary(sa.func._ibis_sqlite_string_ascii),
        ops.Capitalize: unary(sa.func._ibis_sqlite_capitalize),
        ops.Translate: fixed_arity(sa.func._ibis_sqlite_translate, 3),
        ops.Sqrt: unary(sa.func._ibis_sqlite_sqrt),
        ops.Power: fixed_arity(sa.func._ibis_sqlite_power, 2),
        ops.Exp: unary(sa.func._ibis_sqlite_exp),
        ops.Ln: unary(sa.func._ibis_sqlite_ln),
        ops.Log: _log,
        ops.Log10: unary(sa.func._ibis_sqlite_log10),
        ops.Log2: unary(sa.func._ibis_sqlite_log2),
        ops.Floor: unary(sa.func._ibis_sqlite_floor),
        ops.Ceil: unary(sa.func._ibis_sqlite_ceil),
        ops.Sign: unary(sa.func._ibis_sqlite_sign),
        ops.FloorDivide: fixed_arity(sa.func._ibis_sqlite_floordiv, 2),
        ops.Modulus: fixed_arity(sa.func._ibis_sqlite_mod, 2),
        ops.Variance: variance_reduction('_ibis_sqlite_var'),
        ops.StandardDev: toolz.compose(
            sa.func._ibis_sqlite_sqrt, variance_reduction('_ibis_sqlite_var')
        ),
        ops.RowID: lambda *_: sa.literal_column('rowid'),
        ops.Cot: unary(sa.func._ibis_sqlite_cot),
        ops.Cos: unary(sa.func._ibis_sqlite_cos),
        ops.Sin: unary(sa.func._ibis_sqlite_sin),
        ops.Tan: unary(sa.func._ibis_sqlite_tan),
        ops.Acos: unary(sa.func._ibis_sqlite_acos),
        ops.Asin: unary(sa.func._ibis_sqlite_asin),
        ops.Atan: unary(sa.func._ibis_sqlite_atan),
        ops.Atan2: fixed_arity(sa.func._ibis_sqlite_atan2, 2),
        ops.BitOr: reduction(sa.func._ibis_sqlite_bit_or),
        ops.BitAnd: reduction(sa.func._ibis_sqlite_bit_and),
        ops.BitXor: reduction(sa.func._ibis_sqlite_bit_xor),
        ops.Degrees: unary(sa.func._ibis_sqlite_degrees),
        ops.Radians: unary(sa.func._ibis_sqlite_radians),
        ops.Clip: _clip(min_func=sa.func.min, max_func=sa.func.max),
        # sqlite doesn't implement a native xor operator
        ops.BitwiseXor: fixed_arity(sa.func._ibis_sqlite_xor, 2),
        ops.BitwiseNot: unary(sa.func._ibis_sqlite_inv),
    }
)
