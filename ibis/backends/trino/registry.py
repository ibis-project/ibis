import sqlalchemy as sa

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.registry import (
    reduction,
    sqlalchemy_operation_registry,
    unary,
)
from ibis.backends.postgres.registry import _corr, _covar

operation_registry = sqlalchemy_operation_registry.copy()

# TODO: trino doesn't support `& |` for bitwise ops, it wants `bitwise_and` and `bitwise_or``


operation_registry.update(
    {
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
    }
)
