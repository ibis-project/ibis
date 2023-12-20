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
    dtype = op.dtype
    value = op.value

    if value is None:
        return sa.null()
    elif (
        # handle UUIDs in sqlalchemy < 2
        vparse(sa.__version__) < vparse("2") and dtype.is_uuid()
    ):
        return sa.literal(str(value), type_=t.get_sqla_type(dtype))
    elif dtype.is_timestamp():
        if dtype.timezone is not None:
            return sa.func.to_utc_timestamp_tz(value.isoformat(timespec="microseconds"))
        return sa.func.to_timestamp(
            # comma for sep here because T is a special character in Oracle
            # the FX prefix means "requires an exact match"
            value.isoformat(sep=",", timespec="microseconds"),
            "FXYYYY-MM-DD,HH24:MI:SS.FF6",
        )
    elif dtype.is_date():
        return sa.func.to_date(value.isoformat(), "FXYYYY-MM-DD")
    elif dtype.is_time():
        raise NotImplementedError("Time values are not supported in Oracle")
    return _alchemy_literal(t, op)


def _second(t, op):
    # Oracle returns fractional seconds, so `floor` the result to match
    # the behavior of other backends
    return sa.func.floor(sa.extract("SECOND", t.translate(op.arg)))


def _string_join(t, op):
    sep = t.translate(op.sep)
    values = list(map(t.translate, op.arg))
    return sa.func.concat(*toolz.interpose(sep, values))


def _median(t, op):
    arg = op.arg
    if (where := op.where) is not None:
        arg = ops.IfElse(where, arg, None)

    if arg.dtype.is_numeric():
        return sa.func.median(t.translate(arg))
    return sa.cast(
        sa.func.percentile_disc(0.5).within_group(t.translate(arg)),
        t.get_sqla_type(op.dtype),
    )


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
        ops.Median: _median,
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
