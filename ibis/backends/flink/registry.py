import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import ExprTranslator
from ibis.backends.base.sql.registry import helpers
from ibis.backends.base.sql.registry import (
    operation_registry as base_operation_registry,
)
from ibis.common.temporal import TimestampUnit

operation_registry = base_operation_registry.copy()


def _count_star(translator: ExprTranslator, op: ops.Node) -> str:
    return "count(*)"


def _timestamp_from_unix(translator, op):
    arg, unit = op.args

    if unit == TimestampUnit.MILLISECOND:
        return f"TO_TIMESTAMP_LTZ({helpers.quote_identifier(arg.name, force=True)}, 3)"
    elif unit == TimestampUnit.SECOND:
        return f"TO_TIMESTAMP_LTZ({helpers.quote_identifier(arg.name, force=True)}, 0)"
    raise ValueError(f"{unit!r} unit is not supported!")


operation_registry.update(
    {
        ops.CountStar: _count_star,
        ops.TimestampFromUNIX: _timestamp_from_unix,
    }
)
