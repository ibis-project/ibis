from __future__ import annotations

import sqlalchemy as sa

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.registry import _literal as base_literal
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    sqlalchemy_operation_registry,
    unary,
)


def _literal(translator, op):
    dtype = op.output_dtype
    if dtype.is_decimal():
        value = op.value
        if value.is_nan():
            return sa.literal_column("CAST('NaN' AS DOUBLE)")
        if value.is_infinite():
            prefix = "-" * value.is_signed()
            return sa.literal_column(f"CAST('{prefix}Infinity' AS DOUBLE)")
    return base_literal(translator, op)


def _correlation(t, op):
    # TODO: sample vs pop
    x, y = op.left, op.right
    if (x_type := x.output_dtype).is_boolean():
        # Dremio can't cast bool.
        ty = dt.Int32(nullable=x_type.nullable)
        # XXX: would be better to implement in cast translation
        x = ops.SimpleCase(
            x,
            (ops.Literal(True, dt.Boolean), ops.Literal(False, dt.Boolean)),
            (ops.Literal(1, ty), ops.Literal(0, ty)),
            ops.NullLiteral(),
        )
    if (y_type := y.output_dtype).is_boolean():
        ty = dt.Int32(nullable=y_type.nullable)
        y = ops.SimpleCase(
            y,
            (ops.Literal(True, dt.Boolean), ops.Literal(False, dt.Boolean)),
            (ops.Literal(1, ty), ops.Literal(0, ty)),
            ops.NullLiteral(),
        )
    lhs = t.translate(x)
    rhs = t.translate(y)
    return sa.sql.functions.Function(sa.sql.quoted_name("corr", '"'), lhs, rhs)


def _covariance(t, op):
    if op.how == 'pop':
        func = sa.func.covar_pop
    elif op.how == 'sample':
        func = sa.func.covar_samp
    else:
        raise ValueError(op.how)

    x, y = op.left, op.right
    if (x_type := x.output_dtype).is_boolean():
        # Dremio can't cast bool.
        ty = dt.Int32(nullable=x_type.nullable)
        x = ops.SimpleCase(
            x,
            (ops.Literal(True, dt.Boolean), ops.Literal(False, dt.Boolean)),
            (ops.Literal(1, ty), ops.Literal(0, ty)),
            ops.NullLiteral(),
        )
    if (y_type := y.output_dtype).is_boolean():
        ty = dt.Int32(nullable=y_type.nullable)
        y = ops.SimpleCase(
            y,
            (ops.Literal(True, dt.Boolean), ops.Literal(False, dt.Boolean)),
            (ops.Literal(1, ty), ops.Literal(0, ty)),
            ops.NullLiteral(),
        )
    lhs = t.translate(x)
    rhs = t.translate(y)
    return func(lhs, rhs)


operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(
    {
        # Literal
        ops.Literal: _literal,
        # Math
        ops.BitwiseLeftShift: fixed_arity(sa.func.lshift, 2),
        ops.BitwiseRightShift: fixed_arity(sa.func.rshift, 2),
        ops.BitwiseXor: fixed_arity(sa.func.xor, 2),
        ops.Log: fixed_arity(lambda a, b: sa.func.log(b, a), 2),
        ops.Log2: unary(lambda arg: sa.func.log(2, arg)),
        ops.Log10: unary(sa.func.log10),
        # Reductions
        ops.Correlation: _correlation,
        ops.Covariance: _covariance,
        # Types
        ops.TypeOf: unary(sa.func.typeof),
    }
)
