from __future__ import annotations

from typing import TYPE_CHECKING

import ibis.expr.operations as ops
from ibis.backends.base.sql.registry import helpers
from ibis.backends.base.sql.registry import (
    operation_registry as base_operation_registry,
)
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
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.Where: _filter,
    }
)
