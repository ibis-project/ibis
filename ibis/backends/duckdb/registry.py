import collections
import operator

import numpy as np
import sqlalchemy as sa

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import to_sqla_type, unary
from ibis.backends.base.sql.alchemy.registry import (
    _geospatial_functions,
    _table_column,
)
from ibis.backends.postgres.registry import fixed_arity, operation_registry

operation_registry = {
    op: operation_registry[op]
    # duckdb does not support geospatial operations, but shares most of the
    # remaining postgres rules
    for op in operation_registry.keys() - _geospatial_functions.keys()
}


def _round(t, expr):
    arg, digits = expr.op().args
    sa_arg = t.translate(arg)

    if digits is None:
        return sa.func.round(sa_arg)

    return sa.func.round(sa_arg, t.translate(digits))


_LOG_BASE_FUNCS = {
    2: sa.func.log2,
    10: sa.func.log,
}


def _generic_log(arg, base):
    return sa.func.ln(arg) / sa.func.ln(base)


def _log(t, expr):
    arg, base = expr.op().args
    sa_arg = t.translate(arg)
    if base is not None:
        sa_base = t.translate(base)
        try:
            base_value = sa_base.value
        except AttributeError:
            return _generic_log(sa_arg, sa_base)
        else:
            func = _LOG_BASE_FUNCS.get(base_value, _generic_log)
            return func(sa_arg)
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


def _literal(_, expr):
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
                    for i, val in enumerate(value.values())
                )
            )
        raise NotImplementedError(
            f"Ibis dtype `{dtype}` with mapping type "
            f"`{type(value).__name__}` isn't yet supported with the duckdb "
            "backend"
        )
    return sa.cast(sa.literal(value), sqla_type)


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


def _regex_extract(t, expr):
    string, pattern, index = map(t.translate, expr.op().args)
    result = sa.case(
        [
            (
                sa.func.regexp_matches(string, pattern),
                sa.func.regexp_extract(
                    string,
                    pattern,
                    # DuckDB requires the index to be a constant so we compile
                    # the value and inline it using sa.text
                    sa.text(
                        str(
                            (index + 1).compile(
                                compile_kwargs=dict(literal_binds=True)
                            )
                        )
                    ),
                ),
            )
        ],
        else_="",
    )
    return result


operation_registry.update(
    {
        ops.ArrayColumn: _array_column,
        ops.ArrayConcat: fixed_arity(sa.func.array_concat, 2),
        ops.DayOfWeekName: unary(sa.func.dayname),
        ops.Literal: _literal,
        ops.Log2: unary(sa.func.log2),
        ops.Ln: unary(sa.func.ln),
        ops.Log: _log,
        # TODO: map operations, but DuckDB's maps are multimaps
        ops.Modulus: fixed_arity(operator.mod, 2),
        ops.Round: _round,
        ops.StructField: _struct_field,
        ops.TableColumn: _table_column,
        ops.TimestampDiff: fixed_arity(sa.func.age, 2),
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.Translate: fixed_arity(sa.func.replace, 3),
        ops.TimestampNow: fixed_arity(sa.func.now, 0),
        ops.RegexExtract: _regex_extract,
        ops.RegexReplace: fixed_arity(sa.func.regexp_replace, 3),
        ops.StringContains: fixed_arity(sa.func.contains, 2),
    }
)

try:
    import duckdb
except ImportError:  # pragma: no cover
    pass
else:
    from packaging.version import parse as vparse

    # 0.3.2 has zero-based array indexing, 0.3.3 has one-based array indexing
    #
    # 0.3.2: we pass in the user's arguments unchanged
    # 0.3.3: use the postgres implementation which is also one-based
    if vparse(duckdb.__version__) < vparse("0.3.3"):  # pragma: no cover
        operation_registry[ops.ArrayIndex] = fixed_arity("list_element", 2)

    # don't export these
    del duckdb, vparse  # pragma: no cover
