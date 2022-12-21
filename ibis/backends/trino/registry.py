import sqlalchemy as sa

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    reduction,
    sqlalchemy_operation_registry,
    unary,
)
from ibis.backends.postgres.registry import _corr, _covar

operation_registry = sqlalchemy_operation_registry.copy()

# TODO: trino doesn't support `& |` for bitwise ops, it wants `bitwise_and` and `bitwise_or``

def _arbitrary(t, op):
    if op.how == "heavy":
        raise ValueError('Trino does not support how="heavy"')
    return reduction(sa.func.arbitrary)(t, op)


def _json_get_item(t, op):
    arg = t.translate(op.arg)
    index = t.translate(op.index)
    fmt = "%d" if op.index.output_dtype.is_integer() else '"%s"'
    return sa.func.json_extract(arg, sa.func.format(f"$[{fmt}]", index))


operation_registry.update(
    {
        # conditional expressions
        # static checks are not happy with using "if" as a property
        ops.Where: fixed_arity(getattr(sa.func, 'if'), 3),
        # boolean reductions
        ops.Any: unary(sa.func.bool_or),
        ops.All: unary(sa.func.bool_and),
        ops.NotAny: unary(lambda x: sa.not_(sa.func.bool_or(x))),
        ops.NotAll: unary(lambda x: sa.not_(sa.func.bool_and(x))),
        ops.ArgMin: reduction(sa.func.min_by),
        ops.ArgMax: reduction(sa.func.max_by),
        # array ops
        ops.Correlation: _corr,
        ops.Covariance: _covar,
        ops.ExtractMillisecond: unary(sa.func.millisecond),
        ops.Arbitrary: _arbitrary,
        ops.BitAnd: reduction(sa.func.bitwise_and_agg),
        ops.BitOr: reduction(sa.func.bitwise_or_agg),
        ops.ArrayCollect: reduction(sa.func.array_agg),
        ops.JSONGetItem: _json_get_item,
    }
)
