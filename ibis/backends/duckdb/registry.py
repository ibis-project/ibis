from __future__ import annotations

import operator
from typing import Any, Mapping

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction

import ibis.expr.operations as ops
from ibis.backends.base.sql import alchemy
from ibis.backends.base.sql.alchemy import unary
from ibis.backends.base.sql.alchemy.datatypes import StructType
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


class struct_pack(GenericFunction):
    def __init__(self, values: Mapping[str, Any], *, type: StructType) -> None:
        super().__init__()
        self.values = values
        self.type = type


@compiles(struct_pack, "duckdb")
def compiles_struct_pack(element, compiler, **kw):
    quote = compiler.preparer.quote
    args = ", ".join(
        "{key} := {value}".format(key=quote(key), value=compiler.process(value, **kw))
        for key, value in element.values.items()
    )
    return f"struct_pack({args})"


def _literal(t, op):
    dtype = op.output_dtype
    sqla_type = t.get_sqla_type(dtype)

    value = op.value
    if dtype.is_interval():
        return sa.literal_column(f"INTERVAL '{value} {dtype.resolution}'")
    elif dtype.is_set() or dtype.is_array():
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
        raise NotImplementedError(
            f"Ibis dtype `{dtype}` with mapping type "
            f"`{type(value).__name__}` isn't yet supported with the duckdb "
            "backend"
        )
    else:
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
    return struct_pack(
        dict(zip(op.names, map(t.translate, op.values))),
        type=t.get_sqla_type(op.output_dtype),
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
        ops.IntervalAdd: fixed_arity(operator.add, 2),
        ops.IntervalSubtract: fixed_arity(operator.sub, 2),
        ops.Capitalize: alchemy.sqlalchemy_operation_registry[ops.Capitalize],
        ops.ArrayStringJoin: fixed_arity(
            lambda sep, arr: sa.func.array_aggr(arr, sa.text("'string_agg'"), sep), 2
        ),
    }
)


_invalid_operations = {
    # ibis.expr.operations.analytic
    ops.CumulativeAll,
    ops.CumulativeAny,
    ops.CumulativeOp,
    ops.NTile,
    # ibis.expr.operations.strings
    ops.Translate,
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
