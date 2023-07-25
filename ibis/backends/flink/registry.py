from __future__ import annotations

from typing import TYPE_CHECKING

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.registry import helpers, window
from ibis.backends.base.sql.registry import (
    operation_registry as base_operation_registry,
)
from ibis.backends.flink.utils import translate_literal
from ibis.common.temporal import TimestampUnit

if TYPE_CHECKING:
    from ibis.backends.base.sql.compiler import ExprTranslator

operation_registry = base_operation_registry.copy()


def _count_star(translator: ExprTranslator, op: ops.Node) -> str:
    return "count(*)"


def _timestamp_from_unix(translator: ExprTranslator, op: ops.Node) -> str:
    arg, unit = op.args

    if unit == TimestampUnit.MILLISECOND:
        return f"TO_TIMESTAMP_LTZ({helpers.quote_identifier(arg.name, force=True)}, 3)"
    elif unit == TimestampUnit.SECOND:
        return f"TO_TIMESTAMP_LTZ({helpers.quote_identifier(arg.name, force=True)}, 0)"
    raise ValueError(f"{unit!r} unit is not supported!")


def _extract_field(sql_attr: str) -> str:
    def extract_field_formatter(translator: ExprTranslator, op: ops.Node) -> str:
        arg = translator.translate(op.args[0])
        if sql_attr == "epochseconds":
            return f"UNIX_SECONDS({arg})"
        else:
            return f"EXTRACT({sql_attr} from {arg})"

    return extract_field_formatter


def _filter(translator: ExprTranslator, op: ops.Node) -> str:
    bool_expr = translator.translate(op.bool_expr)
    true_expr = translator.translate(op.true_expr)
    false_null_expr = translator.translate(op.false_null_expr)

    # [TODO](chloeh13q): It's preferable to use the FILTER syntax instead of CASE WHEN
    # to let the planner do more optimizations to reduce the state size; besides, FILTER
    # is more compliant with the SQL standard.
    # For example,
    # ```
    # COUNT(DISTINCT CASE WHEN flag = 'app' THEN user_id ELSE NULL END) AS app_uv
    # ```
    # is equivalent to
    # ```
    # COUNT(DISTINCT) FILTER (WHERE flag = 'app') AS app_uv
    # ```
    return f"CASE WHEN {bool_expr} THEN {true_expr} ELSE {false_null_expr} END"


def _literal(translator: ExprTranslator, op: ops.Literal) -> str:
    return translate_literal(op)


def _format_window_start(translator: ExprTranslator, boundary):
    if boundary is None:
        return 'UNBOUNDED PRECEDING'

    if isinstance(boundary.value, ops.Literal) and boundary.value.value == 0:
        return "CURRENT ROW"

    value = translator.translate(boundary.value)
    return f'{value} PRECEDING'


def _format_window_end(translator: ExprTranslator, boundary):
    if boundary is None:
        raise com.UnsupportedOperationError(
            "OVER RANGE FOLLOWING windows are not supported in Flink yet"
        )

    value = boundary.value
    if isinstance(value, ops.Cast):
        value = boundary.value.arg
    if isinstance(value, ops.Literal):
        if value.value != 0:
            raise com.UnsupportedOperationError(
                "OVER RANGE FOLLOWING windows are not supported in Flink yet"
            )

    return "CURRENT ROW"


def _format_window_frame(translator: ExprTranslator, func, frame):
    components = []

    if frame.group_by:
        partition_args = ', '.join(map(translator.translate, frame.group_by))
        components.append(f'PARTITION BY {partition_args}')

    (order_by,) = frame.order_by
    if order_by.descending is True:
        raise com.UnsupportedOperationError(
            "Flink only supports windows ordered in ASCENDING mode"
        )
    components.append(f'ORDER BY {translator.translate(order_by)}')

    if frame.start is None and frame.end is None:
        # no-op, default is full sample
        pass
    elif not isinstance(func, translator._forbids_frame_clause):
        # [NOTE] Flink allows
        # "ROWS BETWEEN INTERVAL [...] PRECEDING AND CURRENT ROW"
        # but not
        # "RANGE BETWEEN [...] PRECEDING AND CURRENT ROW",
        # but `.over(rows=(-ibis.interval(...), 0)` is not allowed in Ibis
        if isinstance(frame, ops.RangeWindowFrame):
            if not frame.start.value.output_dtype.is_interval():
                # [TODO] need to expand support for range-based interval windowing on expr
                # side, for now only ibis intervals can be used
                raise com.UnsupportedOperationError(
                    "Data Type mismatch between ORDER BY and RANGE clause"
                )

        start = _format_window_start(translator, frame.start)
        end = _format_window_end(translator, frame.end)

        frame = f'{frame.how.upper()} BETWEEN {start} AND {end}'
        components.append(frame)

    return 'OVER ({})'.format(' '.join(components))


def _window(translator: ExprTranslator, op: ops.Node) -> str:
    frame = op.frame
    if not frame.order_by:
        raise com.UnsupportedOperationError(
            "Flink engine does not support generic window clause with no order by"
        )
    if len(frame.order_by) > 1:
        raise com.UnsupportedOperationError(
            "Windows in Flink can only be ordered by a single time column"
        )

    _unsupported_reductions = translator._unsupported_reductions

    func = op.func.__window_op__

    if isinstance(func, _unsupported_reductions):
        raise com.UnsupportedOperationError(
            f'{type(func)} is not supported in window functions'
        )

    if isinstance(func, ops.CumulativeOp):
        arg = window.cumulative_to_window(translator, func, op.frame)
        return translator.translate(arg)

    if isinstance(frame, ops.RowsWindowFrame):
        if frame.max_lookback is not None:
            raise NotImplementedError(
                'Rows with max lookback is not implemented for SQL-based backends.'
            )

    window_formatted = _format_window_frame(translator, func, frame)

    arg_formatted = translator.translate(func.__window_op__)
    result = f'{arg_formatted} {window_formatted}'

    if isinstance(func, ops.RankBase):
        return f'({result} - 1)'
    else:
        return result


operation_registry.update(
    {
        ops.CountStar: _count_star,
        ops.ExtractYear: _extract_field("year"),  # equivalent to YEAR(date)
        ops.ExtractQuarter: _extract_field("quarter"),  # equivalent to QUARTER(date)
        ops.ExtractMonth: _extract_field("month"),  # equivalent to MONTH(date)
        ops.ExtractWeekOfYear: _extract_field("week"),  # equivalent to WEEK(date)
        ops.ExtractDayOfYear: _extract_field("doy"),  # equivalent to DAYOFYEAR(date)
        ops.ExtractDay: _extract_field("day"),  # equivalent to DAYOFMONTH(date)
        ops.ExtractHour: _extract_field("hour"),  # equivalent to HOUR(timestamp)
        ops.ExtractMinute: _extract_field("minute"),  # equivalent to MINUTE(timestamp)
        ops.ExtractSecond: _extract_field("second"),  # equivalent to SECOND(timestamp)
        ops.Literal: _literal,
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.Where: _filter,
        ops.Window: _window,
    }
)
