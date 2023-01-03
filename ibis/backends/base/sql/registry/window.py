from __future__ import annotations

from operator import add, mul, sub

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir

_map_interval_to_microseconds = {
    'W': 604800000000,
    'D': 86400000000,
    'h': 3600000000,
    'm': 60000000,
    's': 1000000,
    'ms': 1000,
    'us': 1,
    'ns': 0.001,
}


_map_interval_op_to_op = {
    # Literal Intervals have two args, i.e.
    # Literal(1, Interval(value_type=int8, unit='D', nullable=True))
    # Parse both args and multipy 1 * _map_interval_to_microseconds['D']
    ops.Literal: mul,
    ops.IntervalMultiply: mul,
    ops.IntervalAdd: add,
    ops.IntervalSubtract: sub,
}


_cumulative_to_reduction = {
    ops.CumulativeSum: ops.Sum,
    ops.CumulativeMin: ops.Min,
    ops.CumulativeMax: ops.Max,
    ops.CumulativeMean: ops.Mean,
    ops.CumulativeAny: ops.Any,
    ops.CumulativeAll: ops.All,
}


def _replace_interval_with_scalar(op: ops.Value) -> float | ir.FloatingScalar:
    """Replace an interval type or expression with its equivalent numeric scalar.

    Parameters
    ----------
    op
        float or interval expression.
        For example, `ibis.interval(days=1) + ibis.interval(hours=5)`

    Returns
    -------
    preceding
        `float` or `ir.FloatingScalar`, depending on the expr.
    """
    if isinstance(op, ops.Literal):
        unit = getattr(op.output_dtype, "unit", "us")
        try:
            micros = _map_interval_to_microseconds[unit]
            return op.value * micros
        except KeyError:
            raise ValueError(f"Unsupported unit {unit!r}")
    elif op.args and isinstance(op.output_dtype, dt.Interval):
        if len(op.args) > 2:
            raise NotImplementedError("'preceding' argument cannot be parsed.")
        left_arg = _replace_interval_with_scalar(op.args[0])
        right_arg = _replace_interval_with_scalar(op.args[1])
        method = _map_interval_op_to_op[type(op)]
        return method(left_arg, right_arg)
    else:
        raise TypeError(f'input has unknown type {type(op)}')


def cumulative_to_window(translator, op, window):
    klass = _cumulative_to_reduction[type(op)]
    new_op = klass(*op.args)

    try:
        rule = translator._rewrites[type(new_op)]
    except KeyError:
        pass
    else:
        new_op = rule(new_op)

    win = ibis.cumulative_window().group_by(window._group_by).order_by(window._order_by)
    new_expr = an.windowize_function(new_op.to_expr(), win)
    return new_expr.op()


def time_range_to_range_window(_, window):
    # Check that ORDER BY column is a single time column:
    order_by_vars = [x.args[0] for x in window._order_by]
    if len(order_by_vars) > 1:
        raise com.IbisInputError(
            f"Expected 1 order-by variable, got {len(order_by_vars)}"
        )

    order_var = order_by_vars[0]
    timestamp_order_var = ops.Cast(order_var, dt.int64).to_expr()
    window = window._replace(order_by=timestamp_order_var, how='range')

    # Need to change preceding interval expression to scalars
    preceding = window.preceding
    if isinstance(preceding, ir.IntervalScalar):
        new_preceding = _replace_interval_with_scalar(preceding.op())
        window = window._replace(preceding=new_preceding)

    return window


def format_window(translator, op, window):
    components = []

    if window.max_lookback is not None:
        raise NotImplementedError(
            'Rows with max lookback is not implemented for Impala-based backends.'
        )

    if window._group_by:
        partition_args = ', '.join(map(translator.translate, window._group_by))
        components.append(f'PARTITION BY {partition_args}')

    if window._order_by:
        order_args = ', '.join(map(translator.translate, window._order_by))
        components.append(f'ORDER BY {order_args}')

    p, f = window.preceding, window.following

    def _prec(p: int | None) -> str:
        assert p is None or p >= 0

        if p is None:
            prefix = 'UNBOUNDED'
        else:
            if not p:
                return 'CURRENT ROW'
            prefix = str(p)
        return f'{prefix} PRECEDING'

    def _foll(f: int | None) -> str:
        assert f is None or f >= 0

        if f is None:
            prefix = 'UNBOUNDED'
        else:
            if not f:
                return 'CURRENT ROW'
            prefix = str(f)

        return f'{prefix} FOLLOWING'

    if translator._forbids_frame_clause and isinstance(
        op.expr, translator._forbids_frame_clause
    ):
        frame = None
    elif p is not None and f is not None:
        frame = f'{window.how.upper()} BETWEEN {_prec(p)} AND {_foll(f)}'
    elif p is not None:
        if isinstance(p, tuple):
            start, end = p
            frame = '{} BETWEEN {} AND {}'.format(
                window.how.upper(), _prec(start), _prec(end)
            )
        else:
            kind = 'ROWS' if p > 0 else 'RANGE'
            frame = f'{kind} BETWEEN {_prec(p)} AND UNBOUNDED FOLLOWING'
    elif f is not None:
        if isinstance(f, tuple):
            start, end = f
            frame = '{} BETWEEN {} AND {}'.format(
                window.how.upper(), _foll(start), _foll(end)
            )
        else:
            kind = 'ROWS' if f > 0 else 'RANGE'
            frame = f'{kind} BETWEEN UNBOUNDED PRECEDING AND {_foll(f)}'
    else:
        frame = None

    if frame is not None:
        components.append(frame)

    return 'OVER ({})'.format(' '.join(components))


_subtract_one = '({} - 1)'.format


_expr_transforms = {
    ops.RowNumber: _subtract_one,
    ops.DenseRank: _subtract_one,
    ops.MinRank: _subtract_one,
    ops.NTile: _subtract_one,
}


def window(translator, op):
    arg, window = op.args

    _unsupported_reductions = (
        ops.ApproxMedian,
        ops.GroupConcat,
        ops.ApproxCountDistinct,
    )

    if isinstance(arg, _unsupported_reductions):
        raise com.UnsupportedOperationError(
            f'{type(arg)} is not supported in window functions'
        )

    if isinstance(arg, ops.CumulativeOp):
        arg = cumulative_to_window(translator, arg, window)
        return translator.translate(arg)

    # Some analytic functions need to have the expression of interest in
    # the ORDER BY part of the window clause
    if isinstance(arg, translator._require_order_by) and not window._order_by:
        window = window.order_by(arg.args[0])

    # Time ranges need to be converted to microseconds.
    # FIXME(kszucs): avoid the expression roundtrip
    if window.how == 'range':
        time_range_types = (dt.Time, dt.Date, dt.Timestamp)
        if any(
            isinstance(c.output_dtype, time_range_types)
            and c.output_shape.is_columnar()
            for c in window._order_by
        ):
            window = time_range_to_range_window(translator, window)

    window_formatted = format_window(translator, op, window)

    arg_formatted = translator.translate(arg)
    result = f'{arg_formatted} {window_formatted}'

    if type(arg) in _expr_transforms:
        return _expr_transforms[type(arg)](result)
    else:
        return result


def shift_like(name):
    def formatter(translator, op):
        arg, offset, default = op.args

        arg_formatted = translator.translate(arg)

        if default is not None:
            if offset is None:
                offset_formatted = '1'
            else:
                offset_formatted = translator.translate(offset)

            default_formatted = translator.translate(default)

            return '{}({}, {}, {})'.format(
                name, arg_formatted, offset_formatted, default_formatted
            )
        elif offset is not None:
            offset_formatted = translator.translate(offset)
            return f'{name}({arg_formatted}, {offset_formatted})'
        else:
            return f'{name}({arg_formatted})'

    return formatter


def ntile(translator, op):
    return f'ntile({translator.translate(op.buckets)})'
