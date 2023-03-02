from __future__ import annotations

import sqlalchemy as sa

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)

operation_registry = sqlalchemy_operation_registry.copy()

operation_registry.update(sqlalchemy_window_functions_registry)

operation_registry.update(
    {
        ops.BitwiseAnd: fixed_arity(sa.func.bitwise_and, 2),
        ops.BitwiseNot: unary(sa.func.bitwise_complement),
        ops.BitwiseOr: fixed_arity(sa.func.bitwise_or, 2),
        ops.BitwiseXor: fixed_arity(sa.func.bitwise_xor, 2),
        ops.BitwiseLeftShift: fixed_arity(sa.func.bitwise_shift_left, 2),
        ops.BitwiseRightShift: fixed_arity(sa.func.bitwise_shift_right, 2),
    }
)
