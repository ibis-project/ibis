from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction

import ibis.expr.operations as ops
from ibis.backends.base.sql import alchemy
from ibis.backends.base.sql.alchemy import unary
from ibis.backends.base.sql.alchemy.registry import (
    _table_column,
    array_filter,
    array_map,
    geospatial_functions,
    reduction,
    try_cast,
)
from ibis.backends.postgres.registry import (
    _array_index,
    _array_slice,
    fixed_arity,
    operation_registry,
)
from ibis.common.exceptions import UnsupportedOperationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ibis.backends.base.sql.alchemy.datatypes import StructType

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


def _generic_log(arg, base, *, type_):
    return sa.func.ln(arg, type_=type_) / sa.func.ln(base, type_=type_)


def _log(t, op):
    arg, base = op.args
    sqla_type = t.get_sqla_type(op.dtype)
    sa_arg = t.translate(arg)
    if base is not None:
        sa_base = t.translate(base)
        try:
            base_value = sa_base.value
        except AttributeError:
            return _generic_log(sa_arg, sa_base, type_=sqla_type)
        else:
            func = _LOG_BASE_FUNCS.get(base_value, _generic_log)
            return func(sa_arg, type_=sqla_type)
    return sa.func.ln(sa_arg, type_=sqla_type)


def _timestamp_from_unix(t, op):
    arg, unit = op.args
    arg = t.translate(arg)

    if unit.short == "ms":
        return sa.func.epoch_ms(arg)
    elif unit.short == "s":
        return sa.func.to_timestamp(arg)
    else:
        raise UnsupportedOperationError(f"{unit!r} unit is not supported!")


def _timestamp_bucket(t, op):
    arg = t.translate(op.arg)
    interval = t.translate(op.interval)

    origin = sa.literal_column("'epoch'::TIMESTAMP")

    if op.offset is not None:
        origin += t.translate(op.offset)
    return sa.func.time_bucket(interval, arg, origin)


class struct_pack(GenericFunction):
    def __init__(self, values: Mapping[str, Any], *, type: StructType) -> None:
        super().__init__()
        self.values = values
        self.type = type


@compiles(struct_pack, "duckdb")
def compiles_struct_pack(element, compiler, **kw):
    quote = compiler.preparer.quote
    args = ", ".join(
        f"{quote(key)} := {compiler.process(value, **kw)}"
        for key, value in element.values.items()
    )
    return f"struct_pack({args})"


def _literal(t, op):
    dtype = op.dtype
    value = op.value

    if value is None:
        return sa.null()

    sqla_type = t.get_sqla_type(dtype)

    if dtype.is_interval():
        return sa.literal_column(f"INTERVAL '{value} {dtype.resolution}'")
    elif dtype.is_array():
        values = value.tolist() if isinstance(value, np.ndarray) else value
        return sa.cast(sa.func.list_value(*values), sqla_type)
    elif dtype.is_floating():
        if not np.isfinite(value):
            if np.isnan(value):
                value = "NaN"
            else:
                assert np.isinf(value), "value is neither finite, nan nor infinite"
                prefix = "-" * (value < 0)
                value = f"{prefix}Inf"
        return sa.cast(sa.literal(value), sqla_type)
    elif dtype.is_struct():
        return struct_pack(
            {
                key: t.translate(ops.Literal(val, dtype=dtype[key]))
                for key, val in value.items()
            },
            type=sqla_type,
        )
    elif dtype.is_string():
        return sa.literal(value)
    elif dtype.is_map():
        return sa.func.map(
            sa.func.list_value(*value.keys()), sa.func.list_value(*value.values())
        )
    elif dtype.is_date():
        return sa.cast(sa.literal(str(value)), sqla_type)
    else:
        return sa.cast(sa.literal(value), sqla_type)


if_ = getattr(sa.func, "if")


def _neg_idx_to_pos(array, idx):
    arg_length = sa.func.array_length(array)
    return if_(idx < 0, arg_length + sa.func.greatest(idx, -arg_length), idx)


def _regex_extract(string, pattern, index):
    return sa.func.regexp_extract(
        string,
        pattern,
        # DuckDB requires the index to be a constant, so we compile
        # the value and inline it by using sa.text
        sa.text(str(index.compile(compile_kwargs=dict(literal_binds=True)))),
    )


def _json_get_item(left, path):
    # Workaround for https://github.com/duckdb/duckdb/issues/5063
    # In some situations duckdb silently does the wrong thing if
    # the path is parametrized.
    sa_path = sa.text(str(path.compile(compile_kwargs=dict(literal_binds=True))))
    return left.op("->")(sa_path)


def _strftime(t, op):
    if not isinstance(op.format_str, ops.Literal):
        raise UnsupportedOperationError(
            f"DuckDB format_str must be a literal `str`; got {type(op.format_str)}"
        )
    return sa.func.strftime(t.translate(op.arg), t.translate(op.format_str))


def _strptime(t, op):
    if not isinstance(op.format_str, ops.Literal):
        raise UnsupportedOperationError(
            f"DuckDB format_str must be a literal `str`; got {type(op.format_str)}"
        )
    return sa.cast(
        sa.func.strptime(t.translate(op.arg), t.translate(op.format_str)),
        t.get_sqla_type(op.dtype),
    )


def _arbitrary(t, op):
    if (how := op.how) == "heavy":
        raise UnsupportedOperationError(
            f"how={how!r} not supported in the DuckDB backend"
        )
    return t._reduction(getattr(sa.func, how), op)


def _string_agg(t, op):
    if not isinstance(op.sep, ops.Literal):
        raise UnsupportedOperationError(
            "Separator argument to group_concat operation must be a constant"
        )
    agg = sa.func.string_agg(t.translate(op.arg), sa.text(repr(op.sep.value)))
    if (where := op.where) is not None:
        return agg.filter(t.translate(where))
    return agg


def _struct_column(t, op):
    return struct_pack(
        dict(zip(op.names, map(t.translate, op.values))),
        type=t.get_sqla_type(op.dtype),
    )


@compiles(array_map, "duckdb")
def compiles_list_apply(element, compiler, **kw):
    *args, signature, result = map(partial(compiler.process, **kw), element.clauses)
    return f"list_apply({', '.join(args)}, {signature} -> {result})"


def _array_map(t, op):
    return array_map(
        t.translate(op.arg), sa.literal_column(f"({op.param})"), t.translate(op.body)
    )


@compiles(array_filter, "duckdb")
def compiles_list_filter(element, compiler, **kw):
    *args, signature, result = map(partial(compiler.process, **kw), element.clauses)
    return f"list_filter({', '.join(args)}, {signature} -> {result})"


def _array_filter(t, op):
    return array_filter(
        t.translate(op.arg), sa.literal_column(f"({op.param})"), t.translate(op.body)
    )


def _array_intersect(t, op):
    name = "x"
    parameter = ops.Argument(
        name=name, shape=op.left.shape, dtype=op.left.dtype.value_type
    )
    return t.translate(
        ops.ArrayFilter(
            op.left, param=name, body=ops.ArrayContains(op.right, parameter)
        )
    )


def _map_keys(t, op):
    m = t.translate(op.arg)
    return sa.cast(sa.func.json_keys(sa.func.to_json(m)), t.get_sqla_type(op.dtype))


def _is_map_literal(op):
    return isinstance(op, ops.Literal) or (
        isinstance(op, ops.Map)
        and isinstance(op.keys, ops.Literal)
        and isinstance(op.values, ops.Literal)
    )


def _map_values(t, op):
    if not _is_map_literal(arg := op.arg):
        raise UnsupportedOperationError(
            "Extracting values of non-literal maps is not yet supported by DuckDB"
        )
    m_json = sa.func.to_json(t.translate(arg))
    return sa.cast(
        sa.func.json_extract_string(m_json, sa.func.json_keys(m_json)),
        t.get_sqla_type(op.dtype),
    )


def _map_merge(t, op):
    if not (_is_map_literal(op.left) and _is_map_literal(op.right)):
        raise UnsupportedOperationError(
            "Merging non-literal maps is not yet supported by DuckDB"
        )
    left = sa.func.to_json(t.translate(op.left))
    right = sa.func.to_json(t.translate(op.right))
    pairs = sa.func.json_merge_patch(left, right)
    keys = sa.func.json_keys(pairs)
    return sa.cast(
        sa.func.map(keys, sa.func.json_extract_string(pairs, keys)),
        t.get_sqla_type(op.dtype),
    )


def _array_zip(t, op):
    args = tuple(map(t.translate, op.arg))

    i = sa.literal_column("i", type_=sa.INTEGER)
    dtype = op.dtype
    return array_map(
        sa.func.range(1, sa.func.greatest(*map(sa.func.array_length, args)) + 1),
        i,
        struct_pack(
            {
                name: sa.func.list_extract(arg, i)
                for name, arg in zip(dtype.value_type.names, args)
            },
            type=t.get_sqla_type(dtype),
        ),
    )


@compiles(try_cast, "duckdb")
def compiles_try_cast(element, compiler, **kw):
    return "TRY_CAST({} AS {})".format(
        compiler.process(element.clauses.clauses[0], **kw),
        compiler.visit_typeclause(element),
    )


def _try_cast(t, op):
    arg = t.translate(op.arg)
    to = t.get_sqla_type(op.to)
    return try_cast(arg, type_=to)


_temporal_delta = fixed_arity(
    lambda part, start, end: sa.func.date_diff(part, end, start), 3
)


def _to_json_collection(t, op):
    typ = t.get_sqla_type(op.dtype)
    return try_cast(t.translate(op.arg), typ, type_=typ)


operation_registry.update(
    {
        ops.ArrayColumn: (
            lambda t, op: sa.cast(
                sa.func.list_value(*map(t.translate, op.cols)),
                t.get_sqla_type(op.dtype),
            )
        ),
        ops.TryCast: _try_cast,
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
        ops.ArrayMap: _array_map,
        ops.ArrayFilter: _array_filter,
        ops.ArrayContains: fixed_arity(sa.func.list_has, 2),
        ops.ArrayPosition: fixed_arity(
            lambda lst, el: sa.func.list_indexof(lst, el) - 1, 2
        ),
        ops.ArrayDistinct: fixed_arity(
            lambda arg: if_(
                arg.is_(sa.null()),
                sa.null(),
                # append a null if the input array has a null
                sa.func.list_distinct(arg)
                + if_(
                    # list_count doesn't count nulls
                    sa.func.list_count(arg) < sa.func.array_length(arg),
                    sa.func.list_value(sa.null()),
                    sa.func.list_value(),
                ),
            ),
            1,
        ),
        ops.ArraySort: fixed_arity(sa.func.list_sort, 1),
        ops.ArrayRemove: lambda t, op: _array_filter(
            t,
            ops.ArrayFilter(
                op.arg,
                param="x",
                body=ops.NotEquals(
                    ops.Argument(
                        name="x", shape=op.arg.shape, dtype=op.arg.dtype.value_type
                    ),
                    op.other,
                ),
            ),
        ),
        ops.ArrayUnion: lambda t, op: t.translate(
            ops.ArrayDistinct(ops.ArrayConcat((op.left, op.right)))
        ),
        ops.ArrayZip: _array_zip,
        ops.DayOfWeekName: unary(sa.func.dayname),
        ops.Literal: _literal,
        ops.Log2: unary(sa.func.log2),
        ops.Ln: unary(sa.func.ln),
        ops.Log: _log,
        ops.IsNan: unary(sa.func.isnan),
        ops.Modulus: fixed_arity(operator.mod, 2),
        ops.Round: _round,
        ops.StructField: (
            lambda t, op: sa.func.struct_extract(
                t.translate(op.arg),
                sa.text(repr(op.field)),
                type_=t.get_sqla_type(op.dtype),
            )
        ),
        ops.TableColumn: _table_column,
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.TimestampBucket: _timestamp_bucket,
        ops.TimestampNow: fixed_arity(
            # duckdb 0.6.0 changes now to be a timestamp with time zone force
            # it back to the original for backwards compatibility
            lambda *_: sa.cast(sa.func.now(), sa.TIMESTAMP),
            0,
        ),
        ops.RegexExtract: fixed_arity(_regex_extract, 3),
        ops.RegexReplace: fixed_arity(
            lambda *args: sa.func.regexp_replace(*args, sa.text("'g'")), 3
        ),
        ops.RegexSearch: fixed_arity(sa.func.regexp_matches, 2),
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
        ops.RowID: lambda *_: sa.literal_column("rowid"),
        ops.StringToTimestamp: _strptime,
        ops.Quantile: reduction(sa.func.quantile_cont),
        ops.MultiQuantile: reduction(sa.func.quantile_cont),
        ops.TypeOf: unary(sa.func.typeof),
        ops.IntervalAdd: fixed_arity(operator.add, 2),
        ops.IntervalSubtract: fixed_arity(operator.sub, 2),
        ops.Capitalize: alchemy.sqlalchemy_operation_registry[ops.Capitalize],
        ops.ArrayStringJoin: fixed_arity(
            lambda sep, arr: sa.func.array_aggr(arr, sa.text("'string_agg'"), sep), 2
        ),
        ops.StartsWith: fixed_arity(sa.func.prefix, 2),
        ops.EndsWith: fixed_arity(sa.func.suffix, 2),
        ops.Argument: lambda _, op: sa.literal_column(op.name),
        ops.Unnest: unary(sa.func.unnest),
        ops.MapGet: fixed_arity(
            lambda arg, key, default: sa.func.coalesce(
                sa.func.list_extract(sa.func.element_at(arg, key), 1), default
            ),
            3,
        ),
        ops.Map: fixed_arity(sa.func.map, 2),
        ops.MapContains: fixed_arity(
            lambda arg, key: sa.func.array_length(sa.func.element_at(arg, key)) != 0, 2
        ),
        ops.MapLength: unary(sa.func.cardinality),
        ops.MapKeys: unary(sa.func.map_keys),
        ops.MapValues: unary(sa.func.map_values),
        ops.MapMerge: fixed_arity(sa.func.map_concat, 2),
        ops.Hash: unary(sa.func.hash),
        ops.Median: reduction(sa.func.median),
        ops.First: reduction(sa.func.first),
        ops.Last: reduction(sa.func.last),
        ops.ArrayIntersect: _array_intersect,
        ops.TimeDelta: _temporal_delta,
        ops.DateDelta: _temporal_delta,
        ops.TimestampDelta: _temporal_delta,
        ops.ToJSONMap: _to_json_collection,
        ops.ToJSONArray: _to_json_collection,
    }
)


_invalid_operations = {
    # ibis.expr.operations.strings
    ops.Translate,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
