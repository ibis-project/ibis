from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction

import ibis.backends.base.sql.registry.geospatial as geo
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
from ibis.backends.duckdb.datatypes import Geometry_WKB
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


def _centroid(t, op):
    arg = t.translate(op.arg)
    return sa.func.st_centroid(arg, type_=Geometry_WKB)


def _geo_flip_coordinates(t, op):
    arg = t.translate(op.arg)
    return sa.func.st_flipcoordinates(arg, type_=Geometry_WKB)


def _geo_end_point(t, op):
    arg = t.translate(op.arg)
    return sa.func.st_endpoint(arg, type_=Geometry_WKB)


def _geo_start_point(t, op):
    arg = t.translate(op.arg)
    return sa.func.st_startpoint(arg, type_=Geometry_WKB)


def _envelope(t, op):
    arg = t.translate(op.arg)
    return sa.func.st_envelope(arg, type_=Geometry_WKB)


def _geo_buffer(t, op):
    arg = t.translate(op.arg)
    radius = t.translate(op.radius)
    return sa.func.st_buffer(arg, radius, type_=Geometry_WKB)


def _geo_unary_union(t, op):
    arg = t.translate(op.arg)
    return sa.func.st_union_agg(arg, type_=Geometry_WKB)


def _geo_point(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return sa.func.st_point(left, right, type_=Geometry_WKB)


def _geo_difference(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return sa.func.st_difference(left, right, type_=Geometry_WKB)


def _geo_intersection(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return sa.func.st_intersection(left, right, type_=Geometry_WKB)


def _geo_union(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    return sa.func.st_union(left, right, type_=Geometry_WKB)


def _geo_convert(t, op):
    arg = t.translate(op.arg)
    source = op.source
    target = op.target

    # sa.true() setting always_xy=True
    return sa.func.st_transform(arg, source, target, sa.true(), type_=Geometry_WKB)


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
        return (
            sa.null() if dtype.is_null() else sa.cast(sa.null(), t.get_sqla_type(dtype))
        )

    sqla_type = t.get_sqla_type(dtype)

    if dtype.is_interval():
        return getattr(sa.func, f"to_{dtype.unit.plural}")(value)
    elif dtype.is_geospatial():
        return sa.literal_column(geo.translate_literal(op, inline_metadata=True))
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
    elif dtype.is_timestamp():
        return sa.cast(sa.literal(value.isoformat()), t.get_sqla_type(dtype))
    elif dtype.is_date():
        return sa.func.make_date(value.year, value.month, value.day)
    elif dtype.is_time():
        return sa.func.make_time(
            value.hour, value.minute, value.second + value.microsecond / 1e6
        )
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
            op.left, param=parameter.param, body=ops.ArrayContains(op.right, parameter)
        )
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


def _array_remove(t, op):
    arg = op.arg
    param = ops.Argument(name="x", shape=arg.shape, dtype=arg.dtype.value_type)
    return _array_filter(
        t,
        ops.ArrayFilter(arg, param=param.param, body=ops.NotEquals(param, op.other)),
    )


def _hexdigest(translator, op):
    how = op.how

    arg_formatted = translator.translate(op.arg)
    if how in ("md5", "sha256"):
        return getattr(sa.func, how)(arg_formatted)
    else:
        raise NotImplementedError(how)


operation_registry.update(
    {
        ops.Array: (
            lambda t, op: sa.cast(
                sa.func.list_value(*map(t.translate, op.exprs)),
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
        ops.ArrayRemove: _array_remove,
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
        ops.Quantile: lambda t, op: (
            reduction(sa.func.quantile_cont)(t, op)
            if op.arg.dtype.is_numeric()
            else reduction(sa.func.quantile_disc)(t, op)
        ),
        ops.MultiQuantile: lambda t, op: (
            reduction(sa.func.quantile_cont)(t, op)
            if op.arg.dtype.is_numeric()
            else reduction(sa.func.quantile_disc)(t, op)
        ),
        ops.TypeOf: unary(sa.func.typeof),
        ops.IntervalAdd: fixed_arity(operator.add, 2),
        ops.IntervalSubtract: fixed_arity(operator.sub, 2),
        ops.Capitalize: alchemy.sqlalchemy_operation_registry[ops.Capitalize],
        ops.ArrayStringJoin: fixed_arity(
            lambda sep, arr: sa.func.array_aggr(arr, sa.text("'string_agg'"), sep), 2
        ),
        ops.StartsWith: fixed_arity(sa.func.prefix, 2),
        ops.EndsWith: fixed_arity(sa.func.suffix, 2),
        ops.Argument: lambda _, op: sa.literal_column(op.param),
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
        ops.HexDigest: _hexdigest,
        ops.Median: reduction(sa.func.median),
        ops.First: reduction(sa.func.first),
        ops.Last: reduction(sa.func.last),
        ops.ArrayIntersect: _array_intersect,
        ops.TimeDelta: _temporal_delta,
        ops.DateDelta: _temporal_delta,
        ops.TimestampDelta: _temporal_delta,
        ops.ToJSONMap: _to_json_collection,
        ops.ToJSONArray: _to_json_collection,
        ops.ArrayFlatten: unary(sa.func.flatten),
        ops.IntegerRange: fixed_arity(sa.func.range, 3),
        # geospatial
        ops.GeoPoint: _geo_point,
        ops.GeoAsText: unary(sa.func.ST_AsText),
        ops.GeoArea: unary(sa.func.ST_Area),
        ops.GeoBuffer: _geo_buffer,
        ops.GeoCentroid: _centroid,
        ops.GeoContains: fixed_arity(sa.func.ST_Contains, 2),
        ops.GeoCovers: fixed_arity(sa.func.ST_Covers, 2),
        ops.GeoCoveredBy: fixed_arity(sa.func.ST_CoveredBy, 2),
        ops.GeoCrosses: fixed_arity(sa.func.ST_Crosses, 2),
        ops.GeoDifference: _geo_difference,
        ops.GeoDisjoint: fixed_arity(sa.func.ST_Disjoint, 2),
        ops.GeoDistance: fixed_arity(sa.func.ST_Distance, 2),
        ops.GeoDWithin: fixed_arity(sa.func.ST_DWithin, 3),
        ops.GeoEndPoint: _geo_end_point,
        ops.GeoEnvelope: _envelope,
        ops.GeoEquals: fixed_arity(sa.func.ST_Equals, 2),
        ops.GeoGeometryType: unary(sa.func.ST_GeometryType),
        ops.GeoIntersection: _geo_intersection,
        ops.GeoIntersects: fixed_arity(sa.func.ST_Intersects, 2),
        ops.GeoIsValid: unary(sa.func.ST_IsValid),
        ops.GeoLength: unary(sa.func.ST_Length),
        ops.GeoNPoints: unary(sa.func.ST_NPoints),
        ops.GeoOverlaps: fixed_arity(sa.func.ST_Overlaps, 2),
        ops.GeoStartPoint: _geo_start_point,
        ops.GeoTouches: fixed_arity(sa.func.ST_Touches, 2),
        ops.GeoUnion: _geo_union,
        ops.GeoUnaryUnion: _geo_unary_union,
        ops.GeoWithin: fixed_arity(sa.func.ST_Within, 2),
        ops.GeoX: unary(sa.func.ST_X),
        ops.GeoY: unary(sa.func.ST_Y),
        ops.GeoConvert: _geo_convert,
        ops.GeoFlipCoordinates: _geo_flip_coordinates,
        # other ops
        ops.TimestampRange: fixed_arity(sa.func.range, 3),
        ops.RegexSplit: fixed_arity(sa.func.str_split_regex, 2),
    }
)


_invalid_operations = {
    # ibis.expr.operations.strings
    ops.Translate,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
