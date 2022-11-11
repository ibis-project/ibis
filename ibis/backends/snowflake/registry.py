import numpy as np
import sqlalchemy as sa

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.registry import fixed_arity, reduction
from ibis.backends.postgres.registry import _literal as _postgres_literal
from ibis.backends.postgres.registry import operation_registry as _operation_registry

operation_registry = _operation_registry.copy()


def _literal(t, op):
    if isinstance(op, ops.Literal) and isinstance(op.output_dtype, dt.Floating):
        value = op.value

        if np.isnan(value):
            return _SF_NAN

        if np.isinf(value):
            return _SF_NEG_INF if value < 0 else _SF_POS_INF
    return _postgres_literal(t, op)


def _string_find(t, op):
    args = [t.translate(op.substr), t.translate(op.arg)]
    if (start := op.start) is not None:
        args.append(t.translate(start) + 1)
    return sa.func.position(*args) - 1


def _round(t, op):
    args = [t.translate(op.arg)]
    if (digits := op.digits) is not None:
        args.append(t.translate(digits))
    return sa.func.round(*args)


def _random(t, op):
    min_value = sa.cast(0, sa.dialects.postgresql.FLOAT())
    max_value = sa.cast(1, sa.dialects.postgresql.FLOAT())
    return sa.func.uniform(min_value, max_value, sa.func.random())


_SF_POS_INF = sa.cast(sa.literal("Inf"), sa.FLOAT)
_SF_NEG_INF = -_SF_POS_INF
_SF_NAN = sa.cast(sa.literal("NaN"), sa.FLOAT)


operation_registry.update(
    {
        ops.JSONGetItem: fixed_arity(sa.func.get, 2),
        ops.StructField: fixed_arity(sa.func.get, 2),
        ops.StringFind: _string_find,
        ops.MapKeys: fixed_arity(sa.func.object_keys, 1),
        ops.BitwiseLeftShift: fixed_arity(sa.func.bitshiftleft, 2),
        ops.BitwiseRightShift: fixed_arity(sa.func.bitshiftright, 2),
        ops.Ln: fixed_arity(sa.func.ln, 1),
        ops.Log2: fixed_arity(lambda arg: sa.func.log(2, arg), 1),
        ops.Log10: fixed_arity(lambda arg: sa.func.log(10, arg), 1),
        ops.Log: fixed_arity(lambda arg, base: sa.func.log(base, arg), 2),
        ops.IsInf: fixed_arity(lambda arg: arg.in_((_SF_POS_INF, _SF_NEG_INF)), 1),
        ops.IsNan: fixed_arity(lambda arg: arg == _SF_NAN, 1),
        ops.Literal: _literal,
        ops.Round: _round,
        ops.Modulus: fixed_arity(sa.func.mod, 2),
        ops.Mode: reduction(sa.func.mode),
        # numbers
        ops.RandomScalar: _random,
        # time and dates
        ops.TimeFromHMS: fixed_arity(sa.func.time_from_parts, 3),
    }
)
