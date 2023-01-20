from __future__ import annotations

import collections
import numbers
import operator

import numpy as np
import sqlalchemy as sa

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import unary
from ibis.backends.base.sql.alchemy.registry import (
    _table_column,
    geospatial_functions,
    reduction,
)
from ibis.backends.postgres.registry import (
    _array_index,
    _array_slice,
    fixed_arity,
    operation_registry,
)

operation_registry = {
    op: operation_registry[op]
    for op in operation_registry.keys() - geospatial_functions.keys()
}


def _round(t, op):
    arg, digits = op.args
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


def _log(t, op):
    arg, base = op.args
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


def _timestamp_from_unix(t, op):
    arg, unit = op.args
    arg = t.translate(arg)

    if unit == "ms":
        return sa.func.epoch_ms(arg)
    elif unit == "s":
        return sa.func.to_timestamp(arg)
    else:
        raise ValueError(f"`{unit}` unit is not supported!")


def _literal(t, op):
    dtype = op.output_dtype
    sqla_type = t.get_sqla_type(dtype)
    value = op.value

    if dtype.is_interval():
        return sa.text(f"INTERVAL '{value} {dtype.resolution}'")
    elif dtype.is_set() or (
        isinstance(value, collections.abc.Sequence) and not isinstance(value, str)
    ):
        return sa.cast(sa.func.list_value(*value), sqla_type)
    elif isinstance(value, np.ndarray):
        return sa.cast(sa.func.list_value(*value.tolist()), sqla_type)
    elif isinstance(value, (numbers.Real, np.floating)) and np.isnan(value):
        return sa.cast(sa.literal("NaN"), sqla_type)
    elif isinstance(value, collections.abc.Mapping):
        if dtype.is_struct():
            placeholders = ", ".join(
                f"{key} := :v{i}" for i, key in enumerate(value.keys())
            )
            text = sa.text(f"struct_pack({placeholders})")
            bound_text = text.bindparams(
                *(sa.bindparam(f"v{i:d}", val) for i, val in enumerate(value.values()))
            )
            name = op.name if isinstance(op, ops.Named) else "tmp"
            params = {name: t.get_sqla_type(dtype)}
            return bound_text.columns(**params).scalar_subquery()
        raise NotImplementedError(
            f"Ibis dtype `{dtype}` with mapping type "
            f"`{type(value).__name__}` isn't yet supported with the duckdb "
            "backend"
        )
    return sa.cast(sa.literal(value), sqla_type)


def _neg_idx_to_pos(array, idx):
    if_ = getattr(sa.func, "if")
    arg_length = sa.func.array_length(array)
    return if_(idx < 0, arg_length + sa.func.greatest(idx, -arg_length), idx)


def _regex_extract(string, pattern, index):
    result = sa.case(
        (
            sa.func.regexp_matches(string, pattern),
            sa.func.regexp_extract(
                string,
                pattern,
                # DuckDB requires the index to be a constant so we compile
                # the value and inline it using sa.text
                sa.text(str(index.compile(compile_kwargs=dict(literal_binds=True)))),
            ),
        ),
        else_="",
    )
    return result


def _json_get_item(left, path):
    # Workaround for https://github.com/duckdb/duckdb/issues/5063
    # In some situations duckdb silently does the wrong thing if
    # the path is parametrized.
    sa_path = sa.text(str(path.compile(compile_kwargs=dict(literal_binds=True))))
    return left.op("->")(sa_path)


def _strftime(t, op):
    format_str = op.format_str
    if not isinstance(format_str_op := format_str, ops.Literal):
        raise TypeError(
            f"DuckDB format_str must be a literal `str`; got {type(format_str)}"
        )
    return sa.func.strftime(t.translate(op.arg), sa.text(repr(format_str_op.value)))


def _arbitrary(t, op):
    if (how := op.how) == "heavy":
        raise ValueError(f"how={how!r} not supported in the DuckDB backend")
    return t._reduction(getattr(sa.func, how), op)


def _string_agg(t, op):
    if not isinstance(op.sep, ops.Literal):
        raise TypeError(
            "Separator argument to group_concat operation must be a constant"
        )
    agg = sa.func.string_agg(t.translate(op.arg), sa.text(repr(op.sep.value)))
    if (where := op.where) is not None:
        return agg.filter(t.translate(where))
    return agg


def _struct_column(t, op):
    compile_kwargs = dict(literal_binds=True)
    translated_pairs = (
        (name, t.translate(value).compile(compile_kwargs=compile_kwargs))
        for name, value in zip(op.names, op.values)
    )
    return sa.func.struct_pack(
        *(sa.text(f"{name} := {value}") for name, value in translated_pairs)
    )


operation_registry.update(
    {
        ops.ArrayColumn: (
            lambda t, op: sa.cast(
                sa.func.list_value(*map(t.translate, op.cols)),
                t.get_sqla_type(op.output_dtype),
            )
        ),
        ops.ArrayConcat: fixed_arity(sa.func.array_concat, 2),
        ops.ArrayRepeat: fixed_arity(
            lambda arg, times: sa.func.flatten(
                sa.func.array(
                    sa.select(arg).select_from(sa.func.range(times)).scalar_subquery()
                )
            ),
            2,
        ),
        ops.ArrayLength: unary(sa.func.array_length),
        ops.ArraySlice: _array_slice(
            index_converter=_neg_idx_to_pos,
            array_length=sa.func.array_length,
            func=sa.func.list_slice,
        ),
        ops.ArrayIndex: _array_index(
            index_converter=_neg_idx_to_pos, func=sa.func.list_extract
        ),
        ops.DayOfWeekName: unary(sa.func.dayname),
        ops.Literal: _literal,
        ops.Log2: unary(sa.func.log2),
        ops.Ln: unary(sa.func.ln),
        ops.Log: _log,
        ops.IsNan: unary(sa.func.isnan),
        # TODO: map operations, but DuckDB's maps are multimaps
        ops.Modulus: fixed_arity(operator.mod, 2),
        ops.Round: _round,
        ops.StructField: (
            lambda t, op: sa.func.struct_extract(
                t.translate(op.arg),
                sa.text(repr(op.field)),
                type_=t.get_sqla_type(op.output_dtype),
            )
        ),
        ops.TableColumn: _table_column,
        ops.TimestampDiff: fixed_arity(sa.func.age, 2),
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.TimestampNow: fixed_arity(
            # duckdb 0.6.0 changes now to be a tiemstamp with time zone force
            # it back to the original for backwards compatibility
            lambda *_: sa.cast(sa.func.now(), sa.TIMESTAMP),
            0,
        ),
        ops.RegexExtract: fixed_arity(_regex_extract, 3),
        ops.RegexReplace: fixed_arity(
            lambda *args: sa.func.regexp_replace(*args, "g"), 3
        ),
        ops.StringContains: fixed_arity(sa.func.contains, 2),
        ops.ApproxMedian: reduction(
            # without inline text, duckdb fails with
            # RuntimeError: INTERNAL Error: Invalid PhysicalType for GetTypeIdSize
            lambda arg: sa.func.approx_quantile(arg, sa.text(str(0.5)))
        ),
        ops.ApproxCountDistinct: reduction(sa.func.approx_count_distinct),
        ops.Mode: reduction(sa.func.mode),
        ops.Strftime: _strftime,
        ops.Arbitrary: _arbitrary,
        ops.GroupConcat: _string_agg,
        ops.StructColumn: _struct_column,
        ops.ArgMin: reduction(sa.func.min_by),
        ops.ArgMax: reduction(sa.func.max_by),
        ops.BitwiseXor: fixed_arity(sa.func.xor, 2),
        ops.JSONGetItem: fixed_arity(_json_get_item, 2),
        ops.RowID: lambda *_: sa.literal_column('rowid'),
        ops.StringToTimestamp: fixed_arity(sa.func.strptime, 2),
        ops.Quantile: reduction(sa.func.quantile_cont),
        ops.MultiQuantile: reduction(sa.func.quantile_cont),
        ops.TypeOf: unary(sa.func.typeof),
    }
)


_invalid_operations = {
    # ibis.expr.operations.analytic
    ops.CumulativeAll,
    ops.CumulativeAny,
    ops.CumulativeOp,
    ops.NTile,
    # ibis.expr.operations.strings
    ops.Capitalize,
    ops.Translate,
    # ibis.expr.operations.temporal
    ops.TimestampDiff,
    # ibis.expr.operations.maps
    ops.MapGet,
    ops.MapContains,
    ops.MapKeys,
    ops.MapValues,
    ops.MapMerge,
    ops.MapLength,
    ops.Map,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
