from __future__ import annotations

import sqlalchemy as sa
import toolz
from packaging.version import parse as vparse

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    reduction,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)
from ibis.backends.base.sql.alchemy.registry import (
    _gen_string_find,
)
from ibis.backends.base.sql.alchemy.registry import (
    _literal as _alchemy_literal,
)

operation_registry = sqlalchemy_operation_registry.copy()

operation_registry.update(sqlalchemy_window_functions_registry)


def _cot(t, op):
    arg = t.translate(op.arg)
    return 1.0 / sa.func.tan(arg, type_=t.get_sqla_type(op.arg.dtype))


def _cov(t, op):
    return t._reduction(getattr(sa.func, f"covar_{op.how[:4]}"), op)


def _corr(t, op):
    if op.how == "sample":
        raise ValueError(
            f"{t.__class__.__name__} only implements population correlation "
            "coefficient"
        )
    return t._reduction(sa.func.corr, op)


def _literal(t, op):
    if (
        # handle UUIDs in sqlalchemy < 2
        vparse(sa.__version__) < vparse("2")
        and (dtype := op.dtype).is_uuid()
        and (value := op.value) is not None
    ):
        return sa.literal(str(value), type_=t.get_sqla_type(dtype))
    return _alchemy_literal(t, op)


def _second(t, op):
    # Oracle returns fractional seconds, so `floor` the result to match
    # the behavior of other backends
    return sa.func.floor(sa.extract("SECOND", t.translate(op.arg)))


def _string_join(t, op):
    sep = t.translate(op.sep)
    values = list(map(t.translate, op.arg))
    return sa.func.concat(*toolz.interpose(sep, values))


operation_registry.update(
    {
        ops.Log2: unary(lambda arg: sa.func.log(2, arg)),
        ops.Log10: unary(lambda arg: sa.func.log(10, arg)),
        ops.Log: fixed_arity(lambda arg, base: sa.func.log(base, arg), 2),
        ops.Power: fixed_arity(sa.func.power, 2),
        ops.Cot: _cot,
        ops.Pi: lambda *_: sa.func.ACOS(-1),
        ops.RandomScalar: fixed_arity(sa.func.dbms_random.value, 0),
        ops.Degrees: lambda t, op: 180 * t.translate(op.arg) / t.translate(ops.Pi()),
        ops.Radians: lambda t, op: t.translate(ops.Pi()) * t.translate(op.arg) / 180,
        # Aggregate Functions
        ops.Covariance: _cov,
        ops.Correlation: _corr,
        ops.ApproxMedian: reduction(sa.func.approx_median),
        ops.Median: reduction(sa.func.median),
        # Temporal
        ops.ExtractSecond: _second,
        # String
        ops.StrRight: fixed_arity(lambda arg, nchars: sa.func.substr(arg, -nchars), 2),
        ops.StringJoin: _string_join,
        ops.StringFind: _gen_string_find(sa.func.instr),
        # Generic
        ops.Hash: unary(sa.func.ora_hash),
        ops.Literal: _literal,
        ops.Levenshtein: fixed_arity(sa.func.utl_match.edit_distance, 2),
    }
)

_invalid_operations = set()

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
