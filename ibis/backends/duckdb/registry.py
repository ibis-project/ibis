import collections

import numpy as np
import sqlalchemy as sa

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import to_sqla_type, unary

from ..base.sql.alchemy.registry import _table_column
from ..postgres.registry import fixed_arity, operation_registry

operation_registry = operation_registry.copy()


def _round(t, expr):
    arg, digits = expr.op().args
    sa_arg = t.translate(arg)

    if digits is None:
        return sa.func.round(sa_arg)

    result = sa.func.round(sa_arg, t.translate(digits))
    return result


def _mod(t, expr):
    left, right = map(t.translate, expr.op().args)
    return left % right


def _log(t, expr):
    arg, base = expr.op().args
    sa_arg = t.translate(arg)
    if base is not None:
        sa_base = t.translate(base)
        if sa_base.value == 2:
            return sa.func.log2(sa_arg)
        elif sa_base.value == 10:
            return sa.func.log(sa_arg)
        else:
            raise NotImplementedError
        return sa.func.log(sa_base, sa_arg)
    return sa.func.ln(sa_arg)


def _timestamp_from_unix(t, expr):
    op = expr.op()
    arg, unit = op.args
    arg = t.translate(arg)

    if unit in {"us", "ns"}:
        raise ValueError(f"`{unit}` unit is not supported!")

    if unit == "ms":
        return sa.func.epoch_ms(arg)
    elif unit == "s":
        return sa.func.to_timestamp(arg)


def _literal(t, expr):
    dtype = expr.type()
    sqla_type = to_sqla_type(dtype)
    op = expr.op()
    value = op.value

    if isinstance(dtype, dt.Interval):
        return sa.text(f"INTERVAL '{value} {dtype.resolution}'")
    elif isinstance(dtype, dt.Set) or (
        isinstance(value, collections.abc.Sequence)
        and not isinstance(value, str)
    ):
        return sa.cast(sa.func.list_value(*value), sqla_type)
    elif isinstance(value, np.ndarray):
        return sa.cast(sa.func.list_value(*value.tolist()), sqla_type)
    elif isinstance(value, collections.abc.Mapping):
        if isinstance(dtype, dt.Struct):
            placeholders = ", ".join(
                f"{key!r}: :v{i}" for i, key in enumerate(value.keys())
            )
            return sa.text(f"{{{placeholders}}}").bindparams(
                *(
                    sa.bindparam(f"v{i:d}", val)
                    for i, (key, val) in enumerate(value.items())
                )
            )
        raise NotImplementedError(
            f"Ibis dtype `{dtype}` with mapping type "
            f"`{type(value).__name__}` isn't yet supported with the duckdb "
            "backend"
        )
    return sa.literal(value)


def _array_column(t, expr):
    (arg,) = expr.op().args
    sqla_type = to_sqla_type(expr.type())
    return sa.cast(sa.func.list_value(*map(t.translate, arg)), sqla_type)


def _struct_field(t, expr):
    op = expr.op()
    return sa.func.struct_extract(
        t.translate(op.arg),
        sa.text(repr(op.field)),
        type_=to_sqla_type(expr.type()),
    )


operation_registry.update(
    {
        ops.ArrayColumn: _array_column,
        ops.ArrayConcat: fixed_arity('array_concat', 2),
        ops.DayOfWeekName: unary(sa.func.dayname),
        ops.Literal: _literal,
        ops.Log2: unary(sa.func.log2),
        ops.Log: _log,
        # TODO: map operations, but DuckDB's maps are multimaps
        ops.Modulus: _mod,
        ops.Round: _round,
        ops.StructField: _struct_field,
        ops.TableColumn: _table_column,
        ops.TimestampDiff: fixed_arity('age', 2),
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.Translate: fixed_arity('replace', 3),
    }
)
