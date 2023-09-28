from __future__ import annotations

from typing import TYPE_CHECKING

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.registry import fixed_arity, helpers, unary
from ibis.backends.base.sql.registry import (
    operation_registry as base_operation_registry,
)
from ibis.backends.base.sql.registry.main import varargs
from ibis.common.temporal import TimestampUnit

if TYPE_CHECKING:
    from ibis.backends.base.sql.compiler import ExprTranslator

operation_registry = base_operation_registry.copy()


def _count_star(translator: ExprTranslator, op: ops.Node) -> str:
    if (where := op.where) is not None:
        condition = f" FILTER (WHERE {translator.translate(where)})"
    else:
        condition = ""

    return f"COUNT(*){condition}"


def _date(translator: ExprTranslator, op: ops.Node) -> str:
    (arg,) = op.args
    return f"CAST({translator.translate(arg)} AS DATE)"


def _extract_field(sql_attr: str) -> str:
    def extract_field_formatter(translator: ExprTranslator, op: ops.Node) -> str:
        arg = translator.translate(op.args[0])
        return f"EXTRACT({sql_attr} from {arg})"

    return extract_field_formatter


def _literal(translator: ExprTranslator, op: ops.Literal) -> str:
    from ibis.backends.flink.utils import translate_literal

    return translate_literal(op)


def _try_cast(translator: ExprTranslator, op: ops.Node) -> str:
    arg_formatted = translator.translate(op.arg)

    if op.arg.dtype.is_temporal() and op.to.is_numeric():
        # The cast from TIMESTAMP type to NUMERIC type is not allowed.
        # It's recommended to use UNIX_TIMESTAMP(CAST(timestamp_col AS STRING)) instead.
        return f"UNIX_TIMESTAMP(TRY_CAST({arg_formatted} AS STRING))"
    else:
        sql_type = helpers.type_to_sql_string(op.to)
        return f"TRY_CAST({arg_formatted} AS {sql_type})"


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


def _timestamp_diff(translator: ExprTranslator, op: ops.Node) -> str:
    left = translator.translate(op.left)
    right = translator.translate(op.right)
    return f"timestampdiff(second, {left}, {right})"


def _timestamp_from_unix(translator: ExprTranslator, op: ops.Node) -> str:
    arg, unit = op.args

    numeric = helpers.quote_identifier(arg.name, force=True)
    if unit == TimestampUnit.MILLISECOND:
        precision = 3
    elif unit == TimestampUnit.SECOND:
        precision = 0
    else:
        raise ValueError(f"{unit!r} unit is not supported!")

    return f"TO_TIMESTAMP_LTZ({numeric}, {precision})"


def _format_window_start(translator: ExprTranslator, boundary):
    if boundary is None:
        return "UNBOUNDED PRECEDING"

    if isinstance(boundary.value, ops.Literal) and boundary.value.value == 0:
        return "CURRENT ROW"

    value = translator.translate(boundary.value)
    return f"{value} PRECEDING"


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
        partition_args = ", ".join(map(translator.translate, frame.group_by))
        components.append(f"PARTITION BY {partition_args}")

    (order_by,) = frame.order_by
    if order_by.descending is True:
        raise com.UnsupportedOperationError(
            "Flink only supports windows ordered in ASCENDING mode"
        )
    components.append(f"ORDER BY {translator.translate(order_by)}")

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
            if not frame.start.value.dtype.is_interval():
                # [TODO] need to expand support for range-based interval windowing on expr
                # side, for now only ibis intervals can be used
                raise com.UnsupportedOperationError(
                    "Data Type mismatch between ORDER BY and RANGE clause"
                )

        start = _format_window_start(translator, frame.start)
        end = _format_window_end(translator, frame.end)

        frame = f"{frame.how.upper()} BETWEEN {start} AND {end}"
        components.append(frame)

    return "OVER ({})".format(" ".join(components))


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
            f"{type(func)} is not supported in window functions"
        )

    if isinstance(frame, ops.RowsWindowFrame):
        if frame.max_lookback is not None:
            raise NotImplementedError(
                "Rows with max lookback is not implemented for SQL-based backends."
            )

    window_formatted = _format_window_frame(translator, func, frame)

    arg_formatted = translator.translate(func.__window_op__)
    result = f"{arg_formatted} {window_formatted}"

    if isinstance(func, (ops.RankBase, ops.NTile)):
        return f"({result} - 1)"
    else:
        return result


def _clip(translator: ExprTranslator, op: ops.Node) -> str:
    from ibis.backends.flink.utils import _to_pyflink_types

    arg = translator.translate(op.arg)

    if op.upper is not None:
        upper = translator.translate(op.upper)
        arg = f"IF({arg} > {upper} AND {arg} IS NOT NULL, {upper}, {arg})"

    if op.lower is not None:
        lower = translator.translate(op.lower)
        arg = f"IF({arg} < {lower} AND {arg} IS NOT NULL, {lower}, {arg})"

    return f"CAST({arg} AS {_to_pyflink_types[type(op.dtype)]!s})"


def _floor_divide(translator: ExprTranslator, op: ops.Node) -> str:
    left = translator.translate(op.left)
    right = translator.translate(op.right)
    return f"FLOOR(({left}) / ({right}))"


def _day_of_week_index(translator: ExprTranslator, op: ops.Node) -> str:
    """Convert timestamp to day-of-week integer."""
    arg = op.args[0]
    arg_ = translator.translate(arg)
    return f"MOD(DAYOFWEEK({arg_}) + 5, 7)"


def _string_to_timestamp(translator, op):
    arg_ = translator.translate(op.arg)
    format_string = translator.translate(op.format_str)
    return f"TO_TIMESTAMP({arg_}, {format_string})"


operation_registry.update(
    {
        # Unary operations
        ops.Coalesce: varargs("coalesce"),
        ops.NullIf: fixed_arity("nullif", 2),
        ops.RandomScalar: lambda *_: "rand()",
        ops.Degrees: unary("degrees"),
        ops.Radians: unary("radians"),
        # Unary aggregates
        ops.CountStar: _count_star,
        # String operations
        ops.StringLength: unary("char_length"),
        ops.StrRight: fixed_arity("right", 2),
        ops.RegexSearch: fixed_arity("regexp", 2),
        # Timestamp operations
        ops.Date: _date,
        ops.ExtractYear: _extract_field("year"),  # equivalent to YEAR(date)
        ops.ExtractMonth: _extract_field("month"),  # equivalent to MONTH(date)
        ops.ExtractDay: _extract_field("day"),  # equivalent to DAYOFMONTH(date)
        ops.ExtractQuarter: _extract_field("quarter"),  # equivalent to QUARTER(date)
        ops.ExtractWeekOfYear: _extract_field("week"),  # equivalent to WEEK(date)
        ops.ExtractDayOfYear: _extract_field("doy"),  # equivalent to DAYOFYEAR(date)
        ops.ExtractHour: _extract_field("hour"),  # equivalent to HOUR(timestamp)
        ops.ExtractMinute: _extract_field("minute"),  # equivalent to MINUTE(timestamp)
        ops.ExtractSecond: _extract_field("second"),  # equivalent to SECOND(timestamp)
        ops.ExtractMillisecond: _extract_field("microsecond"),
        ops.ExtractMillisecond: _extract_field("millisecond"),
        # Other operations
        ops.Literal: _literal,
        ops.TryCast: _try_cast,
        ops.IfElse: _filter,
        ops.TimestampDiff: _timestamp_diff,
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.Window: _window,
        ops.Clip: _clip,
        # Binary operations
        ops.Power: fixed_arity("power", 2),
        ops.FloorDivide: _floor_divide,
        # Temporal functions
        ops.DayOfWeekIndex: _day_of_week_index,
        ops.StringToTimestamp: _string_to_timestamp,
    }
)
