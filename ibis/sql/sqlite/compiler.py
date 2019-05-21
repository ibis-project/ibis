# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sqlalchemy as sa

import toolz

import ibis

from ibis.sql.alchemy import unary, varargs, fixed_arity, _variance_reduction

import ibis.sql.alchemy as alch
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.common as com


_operation_registry = alch._operation_registry.copy()


def _cast(t, expr):
    # It's not all fun and games with SQLite

    op = expr.op()
    arg, target_type = op.args
    sa_arg = t.translate(arg)
    sa_type = t.get_sqla_type(target_type)

    if isinstance(target_type, dt.Timestamp):
        if isinstance(arg, ir.IntegerValue):
            return sa.func.datetime(sa_arg, 'unixepoch')
        elif isinstance(arg, ir.StringValue):
            return sa.func.strftime('%Y-%m-%d %H:%M:%f', sa_arg)
        raise com.UnsupportedOperationError(type(arg))

    if isinstance(target_type, dt.Date):
        if isinstance(arg, ir.IntegerValue):
            return sa.func.date(sa.func.datetime(sa_arg, 'unixepoch'))
        elif isinstance(arg, ir.StringValue):
            return sa.func.date(sa_arg)
        raise com.UnsupportedOperationError(type(arg))

    if isinstance(arg, ir.CategoryValue) and target_type == 'int32':
        return sa_arg
    else:
        return sa.cast(sa_arg, sa_type)


def _substr(t, expr):
    f = sa.func.substr

    arg, start, length = expr.op().args

    sa_arg = t.translate(arg)
    sa_start = t.translate(start)

    if length is None:
        return f(sa_arg, sa_start + 1)
    else:
        sa_length = t.translate(length)
        return f(sa_arg, sa_start + 1, sa_length)


def _string_right(t, expr):
    f = sa.func.substr

    arg, length = expr.op().args

    sa_arg = t.translate(arg)
    sa_length = t.translate(length)

    return f(sa_arg, -sa_length, sa_length)


def _string_find(t, expr):
    arg, substr, start, _ = expr.op().args

    if start is not None:
        raise NotImplementedError

    sa_arg = t.translate(arg)
    sa_substr = t.translate(substr)

    f = sa.func.instr
    return f(sa_arg, sa_substr) - 1


def _infix_op(infix_sym):
    def formatter(t, expr):
        op = expr.op()
        left, right = op.args

        left_arg = t.translate(left)
        right_arg = t.translate(right)
        return left_arg.op(infix_sym)(right_arg)

    return formatter


def _strftime(t, expr):
    arg, format = expr.op().args
    sa_arg = t.translate(arg)
    sa_format = t.translate(format)
    return sa.func.strftime(sa_format, sa_arg)


def _strftime_int(fmt):
    def translator(t, expr):
        arg, = expr.op().args
        sa_arg = t.translate(arg)
        return sa.cast(sa.func.strftime(fmt, sa_arg), sa.INTEGER)
    return translator


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
                'Unsupported truncate unit {!r}'.format(unit)
            )
        return func(sa_arg, modifier)

    return translator


def _now(t, expr):
    return sa.func.datetime('now')


def _millisecond(t, expr):
    arg, = expr.op().args
    sa_arg = t.translate(arg)
    fractional_second = sa.func.strftime('%f', sa_arg)
    return (fractional_second * 1000) % 1000


def _identical_to(t, expr):
    left, right = args = expr.op().args
    if left.equals(right):
        return True
    else:
        left, right = map(t.translate, args)
        return sa.func.coalesce(
            (left.is_(None) & right.is_(None)) | (left == right),
            False
        )


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
        f.substr(
            f.quote(
                f.zeroblob((times + 1) / 2)
            ),
            3,
            times
        ),
        '0',
        arg
    )


def _generic_pad(arg, length, pad):
    f = sa.func
    arg_length = f.length(arg)
    pad_length = f.length(pad)
    number_of_zero_bytes = (
        (length - arg_length - 1 + pad_length) / pad_length + 1) / 2
    return f.substr(
        f.replace(
            f.replace(
                f.substr(f.quote(f.zeroblob(number_of_zero_bytes)), 3),
                "'",
                ''
            ),
            '0',
            pad
        ),
        1,
        length - f.length(arg)
    )


def _lpad(t, expr):
    arg, length, pad = map(t.translate, expr.op().args)
    return _generic_pad(arg, length, pad) + arg


def _rpad(t, expr):
    arg, length, pad = map(t.translate, expr.op().args)
    return arg + _generic_pad(arg, length, pad)


_operation_registry.update({
    ops.Cast: _cast,

    ops.Substring: _substr,
    ops.StrRight: _string_right,

    ops.StringFind: _string_find,

    ops.Least: varargs(sa.func.min),
    ops.Greatest: varargs(sa.func.max),
    ops.IfNull: fixed_arity(sa.func.ifnull, 2),

    ops.DateTruncate: _truncate(sa.func.date),
    ops.TimestampTruncate: _truncate(sa.func.datetime),
    ops.Strftime: _strftime,
    ops.ExtractYear: _strftime_int('%Y'),
    ops.ExtractMonth: _strftime_int('%m'),
    ops.ExtractDay: _strftime_int('%d'),
    ops.ExtractHour: _strftime_int('%H'),
    ops.ExtractMinute: _strftime_int('%M'),
    ops.ExtractSecond: _strftime_int('%S'),
    ops.ExtractMillisecond: _millisecond,
    ops.TimestampNow: _now,
    ops.IdenticalTo: _identical_to,

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

    ops.Variance: _variance_reduction('_ibis_sqlite_var'),
    ops.StandardDev: toolz.compose(
        sa.func._ibis_sqlite_sqrt,
        _variance_reduction('_ibis_sqlite_var')
    ),
})


def add_operation(op, translation_func):
    _operation_registry[op] = translation_func


class SQLiteExprTranslator(alch.AlchemyExprTranslator):

    _registry = _operation_registry
    _rewrites = alch.AlchemyExprTranslator._rewrites.copy()
    _type_map = alch.AlchemyExprTranslator._type_map.copy()
    _type_map.update({
        dt.Double: sa.types.REAL,
        dt.Float: sa.types.REAL
    })


rewrites = SQLiteExprTranslator.rewrites
compiles = SQLiteExprTranslator.compiles


class SQLiteDialect(alch.AlchemyDialect):

    translator = SQLiteExprTranslator


dialect = SQLiteDialect


@rewrites(ops.DayOfWeekIndex)
def day_of_week_index(expr):
    return (
        (expr.op().arg.strftime('%w').cast(dt.int16) + 6) % 7
    ).cast(dt.int16)


@rewrites(ops.DayOfWeekName)
def day_of_week_name(expr):
    return (
        expr.op().arg.day_of_week.index()
            .case()
            .when(0, 'Monday')
            .when(1, 'Tuesday')
            .when(2, 'Wednesday')
            .when(3, 'Thursday')
            .when(4, 'Friday')
            .when(5, 'Saturday')
            .when(6, 'Sunday')
            .else_(ibis.NA)
            .end()
    )
