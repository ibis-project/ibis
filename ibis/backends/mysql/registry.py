from __future__ import annotations

import contextlib
import operator

import sqlalchemy as sa

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.backends.base.sql.alchemy.registry import (
    _gen_string_find,
    geospatial_functions,
)

operation_registry = sqlalchemy_operation_registry.copy()

# NOTE: window functions are available from MySQL 8 and MariaDB 10.2
operation_registry.update(sqlalchemy_window_functions_registry)

if geospatial_supported:
    operation_registry.update(geospatial_functions)


_truncate_formats = {
    's': '%Y-%m-%d %H:%i:%s',
    'm': '%Y-%m-%d %H:%i:00',
    'h': '%Y-%m-%d %H:00:00',
    'D': '%Y-%m-%d',
    # 'W': 'week',
    'M': '%Y-%m-01',
    'Y': '%Y-01-01',
}


def _truncate(t, op):
    sa_arg = t.translate(op.arg)
    try:
        fmt = _truncate_formats[op.unit]
    except KeyError:
        raise com.UnsupportedOperationError(f'Unsupported truncate unit {op.unit}')
    return sa.func.date_format(sa_arg, fmt)


def _round(t, op):
    sa_arg = t.translate(op.arg)

    if op.digits is None:
        sa_digits = 0
    else:
        sa_digits = t.translate(op.digits)

    return sa.func.round(sa_arg, sa_digits)


def _interval_from_integer(t, op):
    if op.unit in {'ms', 'ns'}:
        raise com.UnsupportedOperationError(
            f'MySQL does not allow operation with INTERVAL offset {op.unit}'
        )

    sa_arg = t.translate(op.arg)
    text_unit = op.output_dtype.resolution.upper()

    # XXX: Is there a better way to handle this? I.e. can we somehow use
    # the existing bind parameter produced by translate and reuse its name in
    # the string passed to sa.text?
    if isinstance(sa_arg, sa.sql.elements.BindParameter):
        return sa.text(f'INTERVAL :arg {text_unit}').bindparams(arg=sa_arg.value)
    return sa.text(f'INTERVAL {sa_arg} {text_unit}')


def _literal(_, op):
    if op.output_dtype.is_interval():
        if op.output_dtype.unit in {'ms', 'ns'}:
            raise com.UnsupportedOperationError(
                'MySQL does not allow operation '
                f'with INTERVAL offset {op.output_dtype.unit}'
            )
        text_unit = op.output_dtype.resolution.upper()
        sa_text = sa.text(f'INTERVAL :value {text_unit}')
        return sa_text.bindparams(value=op.value)
    elif op.output_dtype.is_set():
        return list(map(sa.literal, op.value))
    else:
        value = op.value
        with contextlib.suppress(AttributeError):
            value = value.to_pydatetime()

        return sa.literal(value)


def _group_concat(t, op):
    if op.where is not None:
        arg = t.translate(ops.Where(op.where, op.arg, ibis.NA))
    else:
        arg = t.translate(op.arg)
    sep = t.translate(op.sep)
    return sa.func.group_concat(arg.op('SEPARATOR')(sep))


def _json_get_item(t, op):
    arg = t.translate(op.arg)
    index = t.translate(op.index)
    if op.index.output_dtype.is_integer():
        path = "$[" + sa.cast(index, sa.TEXT) + "]"
    else:
        path = "$." + index
    return sa.func.json_extract(arg, path)


operation_registry.update(
    {
        ops.Literal: _literal,
        ops.IfNull: fixed_arity(sa.func.ifnull, 2),
        # static checks are not happy with using "if" as a property
        ops.Where: fixed_arity(getattr(sa.func, 'if'), 3),
        # strings
        ops.StringFind: _gen_string_find(sa.func.locate),
        ops.FindInSet: (
            lambda t, op: (
                sa.func.find_in_set(
                    t.translate(op.needle),
                    sa.func.concat_ws(",", *map(t.translate, op.values)),
                )
                - 1
            )
        ),
        # LIKE in mysql is case insensitive
        ops.StartsWith: fixed_arity(
            lambda arg, start: arg.op("LIKE BINARY")(sa.func.concat(start, "%")), 2
        ),
        ops.EndsWith: fixed_arity(
            lambda arg, end: arg.op("LIKE BINARY")(sa.func.concat("%", end)), 2
        ),
        ops.Capitalize: fixed_arity(
            lambda arg: sa.func.concat(
                sa.func.ucase(sa.func.left(arg, 1)), sa.func.substring(arg, 2)
            ),
            1,
        ),
        ops.RegexSearch: fixed_arity(lambda x, y: x.op('REGEXP')(y), 2),
        # math
        ops.Log: fixed_arity(lambda arg, base: sa.func.log(base, arg), 2),
        ops.Log2: unary(sa.func.log2),
        ops.Log10: unary(sa.func.log10),
        ops.Round: _round,
        # dates and times
        ops.DateAdd: fixed_arity(operator.add, 2),
        ops.DateSub: fixed_arity(operator.sub, 2),
        ops.DateDiff: fixed_arity(sa.func.datediff, 2),
        ops.TimestampAdd: fixed_arity(operator.add, 2),
        ops.TimestampSub: fixed_arity(operator.sub, 2),
        ops.TimestampDiff: fixed_arity(
            lambda left, right: sa.func.timestampdiff(sa.text('SECOND'), right, left), 2
        ),
        ops.StringToTimestamp: fixed_arity(
            lambda arg, format_str: sa.func.str_to_date(arg, format_str), 2
        ),
        ops.DateTruncate: _truncate,
        ops.TimestampTruncate: _truncate,
        ops.IntervalFromInteger: _interval_from_integer,
        ops.Strftime: fixed_arity(sa.func.date_format, 2),
        ops.ExtractDayOfYear: unary(sa.func.dayofyear),
        ops.ExtractEpochSeconds: unary(sa.func.UNIX_TIMESTAMP),
        ops.ExtractWeekOfYear: unary(sa.func.weekofyear),
        ops.ExtractMillisecond: fixed_arity(
            lambda arg: sa.func.floor(sa.extract('microsecond', arg) / 1000), 1
        ),
        ops.TimestampNow: fixed_arity(sa.func.now, 0),
        # others
        ops.GroupConcat: _group_concat,
        ops.DayOfWeekIndex: fixed_arity(
            lambda arg: (sa.func.dayofweek(arg) + 5) % 7, 1
        ),
        ops.DayOfWeekName: fixed_arity(lambda arg: sa.func.dayname(arg), 1),
        ops.JSONGetItem: _json_get_item,
    }
)
