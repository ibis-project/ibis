import operator

import pandas as pd
import sqlalchemy as sa

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    to_sqla_type,
    unary,
)
from ibis.backends.base.sql.alchemy.registry import _gen_string_find

operation_registry = sqlalchemy_operation_registry.copy()

# NOTE: window functions are available from MySQL 8 and MariaDB 10.2
operation_registry.update(sqlalchemy_window_functions_registry)


def _capitalize(t, op):
    sa_arg = t.translate(op.arg)
    return sa.func.concat(
        sa.func.ucase(sa.func.left(sa_arg, 1)), sa.func.substring(sa_arg, 2)
    )


def _extract(fmt):
    def translator(t, op):
        sa_arg = t.translate(op.arg)
        if fmt == 'millisecond':
            return sa.extract('microsecond', sa_arg) % 1000
        return sa.extract(fmt, sa_arg)

    return translator


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
        raise com.UnsupportedOperationError(
            f'Unsupported truncate unit {op.unit}'
        )
    return sa.func.date_format(sa_arg, fmt)


def _log(t, op):
    sa_arg = t.translate(op.arg)
    sa_base = t.translate(op.base)
    return sa.func.log(sa_base, sa_arg)


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
            'MySQL does not allow operation '
            'with INTERVAL offset {}'.format(op.unit)
        )

    sa_arg = t.translate(op.arg)
    text_unit = op.output_dtype.resolution.upper()

    # XXX: Is there a better way to handle this? I.e. can we somehow use
    # the existing bind parameter produced by translate and reuse its name in
    # the string passed to sa.text?
    if isinstance(sa_arg, sa.sql.elements.BindParameter):
        return sa.text(f'INTERVAL :arg {text_unit}').bindparams(
            arg=sa_arg.value
        )
    return sa.text(f'INTERVAL {sa_arg} {text_unit}')


def _timestamp_diff(t, op):
    sa_left = t.translate(op.left)
    sa_right = t.translate(op.right)
    return sa.func.timestampdiff(sa.text('SECOND'), sa_right, sa_left)


def _string_to_timestamp(t, op):
    sa_arg = t.translate(op.arg)
    sa_format_str = t.translate(op.format_str)
    if (op.timezone is not None) and op.timezone.value != "UTC":
        raise com.UnsupportedArgumentError(
            'MySQL backend only supports timezone UTC for converting'
            'string to timestamp.'
        )
    return sa.func.str_to_date(sa_arg, sa_format_str)


def _literal(_, op):
    if isinstance(op.output_dtype, dt.Interval):
        if op.output_dtype.unit in {'ms', 'ns'}:
            raise com.UnsupportedOperationError(
                'MySQL does not allow operation '
                f'with INTERVAL offset {op.output_dtype.unit}'
            )
        text_unit = op.output_dtype.resolution.upper()
        sa_text = sa.text(f'INTERVAL :value {text_unit}')
        return sa_text.bindparams(value=op.value)
    elif isinstance(op.output_dtype, dt.Set):
        return list(map(sa.literal, op.value))
    else:
        value = op.value
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()

        lit = sa.literal(value)
        if isinstance(op.output_dtype, dt.Timestamp):
            return sa.cast(lit, to_sqla_type(op.output_dtype))
        return lit


def _group_concat(t, op):
    if op.where is not None:
        # TODO(kszucs): avoid the expression roundtrip
        case = op.where.to_expr().ifelse(op.arg, ibis.NA).op()
        arg = t.translate(case)
    else:
        arg = t.translate(op.arg)
    sep = t.translate(op.sep)
    return sa.func.group_concat(arg.op('SEPARATOR')(sep))


def _day_of_week_index(t, op):
    left = sa.func.dayofweek(t.translate(op.arg)) - 2
    right = 7
    return (left % right + right) % right


def _day_of_week_name(t, op):
    return sa.func.dayname(t.translate(op.arg))


def _find_in_set(t, op):
    return (
        sa.func.find_in_set(
            t.translate(op.needle),
            sa.func.concat_ws(",", *map(t.translate, op.values.values)),
        )
        - 1
    )


operation_registry.update(
    {
        ops.Literal: _literal,
        ops.IfNull: fixed_arity(sa.func.ifnull, 2),
        # strings
        ops.StringFind: _gen_string_find(sa.func.locate),
        ops.FindInSet: _find_in_set,
        ops.Capitalize: _capitalize,
        ops.RegexSearch: fixed_arity(lambda x, y: x.op('REGEXP')(y), 2),
        # math
        ops.Log: _log,
        ops.Log2: unary(sa.func.log2),
        ops.Log10: unary(sa.func.log10),
        ops.Round: _round,
        # dates and times
        ops.DateAdd: fixed_arity(operator.add, 2),
        ops.DateSub: fixed_arity(operator.sub, 2),
        ops.DateDiff: fixed_arity(sa.func.datediff, 2),
        ops.TimestampAdd: fixed_arity(operator.add, 2),
        ops.TimestampSub: fixed_arity(operator.sub, 2),
        ops.TimestampDiff: _timestamp_diff,
        ops.StringToTimestamp: _string_to_timestamp,
        ops.DateTruncate: _truncate,
        ops.TimestampTruncate: _truncate,
        ops.IntervalFromInteger: _interval_from_integer,
        ops.Strftime: fixed_arity(sa.func.date_format, 2),
        ops.ExtractYear: _extract('year'),
        ops.ExtractMonth: _extract('month'),
        ops.ExtractDay: _extract('day'),
        ops.ExtractDayOfYear: unary(sa.func.dayofyear),
        ops.ExtractQuarter: _extract('quarter'),
        ops.ExtractEpochSeconds: unary(sa.func.UNIX_TIMESTAMP),
        ops.ExtractWeekOfYear: unary(sa.func.weekofyear),
        ops.ExtractHour: _extract('hour'),
        ops.ExtractMinute: _extract('minute'),
        ops.ExtractSecond: _extract('second'),
        ops.ExtractMillisecond: _extract('millisecond'),
        ops.TimestampNow: fixed_arity(sa.func.now, 0),
        # others
        ops.GroupConcat: _group_concat,
        ops.DayOfWeekIndex: _day_of_week_index,
        ops.DayOfWeekName: _day_of_week_name,
    }
)
