from __future__ import annotations

import sqlalchemy as sa
import toolz

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)

operation_registry = sqlalchemy_operation_registry.copy()

operation_registry.update(sqlalchemy_window_functions_registry)


def _sign(t, op):
    arg = op.arg
    cond1 = ops.IfElse(ops.Greater(arg, 0), 1, -1)
    cond2 = ops.IfElse(ops.Equals(arg, 0), 0, cond1)
    return t.translate(cond2)


def _join(t, op):
    sep = t.translate(op.sep)
    values = list(map(t.translate, op.arg))
    return sa.func.concat(*toolz.interpose(sep, values))


operation_registry.update(
    {
        ops.BitwiseAnd: fixed_arity(sa.func.bitwise_and, 2),
        ops.BitwiseNot: unary(sa.func.bitwise_complement),
        ops.BitwiseOr: fixed_arity(sa.func.bitwise_or, 2),
        ops.BitwiseXor: fixed_arity(sa.func.bitwise_xor, 2),
        ops.BitwiseLeftShift: fixed_arity(sa.func.bitwise_shift_left, 2),
        ops.BitwiseRightShift: fixed_arity(sa.func.bitwise_shift_right, 2),
        ops.Pi: fixed_arity(lambda: sa.func.acos(-1), 0),
        ops.Modulus: fixed_arity(sa.func.mod, 2),
        ops.Power: fixed_arity(sa.func.power, 2),
        ops.Log10: fixed_arity(sa.func.log10, 1),
        ops.Sign: _sign,
        ops.StringJoin: _join,
        ops.RegexSearch: fixed_arity(sa.func.regexp_like, 2),
    }
)

_invalid_operations = {
    # ibis.expr.operations.generic
    ops.RandomScalar,
    # ibis.expr.operations.strings
    ops.StringAscii,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
