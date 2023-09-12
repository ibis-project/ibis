from __future__ import annotations

from functools import partial, reduce
from typing import Literal

import sqlalchemy as sa
import toolz
from sqlalchemy.ext.compiler import compiles
from trino.sqlalchemy.datatype import DOUBLE

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.registry import (
    _literal as _alchemy_literal,
)
from ibis.backends.base.sql.alchemy.registry import (
    array_filter,
    array_map,
    fixed_arity,
    reduction,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    try_cast,
    unary,
    varargs,
)
from ibis.backends.postgres.registry import _corr, _covar

operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(sqlalchemy_window_functions_registry)


def _array(t, elements):
    return t.translate(ibis.array(elements).op())


def _literal(t, op):
    value = op.value
    dtype = op.dtype

    if value is None:
        return sa.null()

    if dtype.is_struct():
        return sa.cast(sa.func.row(*value.values()), t.get_sqla_type(dtype))
    elif dtype.is_map():
        return sa.func.map(_array(t, value.keys()), _array(t, value.values()))
    elif dtype.is_float64():
        return sa.literal(float(value), type_=DOUBLE())
    elif dtype.is_integer():
        return sa.literal(int(value), type_=t.get_sqla_type(dtype))
    return _alchemy_literal(t, op)


def _arbitrary(t, op):
    if op.how != "first":
        raise com.UnsupportedOperationError(
            'Trino only supports how="first" for `arbitrary` reduction'
        )
    return reduction(sa.func.arbitrary)(t, op)


def _json_get_item(t, op):
    arg = t.translate(op.arg)
    index = t.translate(op.index)
    fmt = "%d" if op.index.dtype.is_integer() else '"%s"'
    return sa.func.json_extract(arg, sa.func.format(f"$[{fmt}]", index))


def _group_concat(t, op):
    if not isinstance(op.sep, ops.Literal):
        raise com.UnsupportedOperationError(
            "Trino group concat separator must be a literal value"
        )

    arg = sa.func.array_agg(t.translate(op.arg))
    if (where := op.where) is not None:
        arg = arg.filter(t.translate(where))
    return sa.func.array_join(arg, t.translate(op.sep))


def _array_column(t, op):
    args = ", ".join(
        str(t.translate(arg).compile(compile_kwargs={"literal_binds": True}))
        for arg in op.cols
    )
    return sa.literal_column(f"ARRAY[{args}]", type_=t.get_sqla_type(op.dtype))


_truncate_precisions = {
    # ms unit is not yet officially documented in Trino's public documentation,
    # but it just works.
    "ms": "millisecond",
    "s": "second",
    "m": "minute",
    "h": "hour",
    "D": "day",
    "W": "week",
    "M": "month",
    "Q": "quarter",
    "Y": "year",
}


def _timestamp_truncate(t, op):
    sa_arg = t.translate(op.arg)
    try:
        precision = _truncate_precisions[op.unit.short]
    except KeyError:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {op.unit!r}")
    return sa.func.date_trunc(precision, sa_arg)


def _timestamp_from_unix(t, op):
    arg, unit = op.args
    arg = t.translate(arg)

    unit_short = unit.short
    if unit_short == "ms":
        try:
            arg //= 1_000
        except TypeError:
            arg = sa.func.floor(arg / 1_000)
        res = sa.func.from_unixtime(arg)
    elif unit_short == "s":
        res = sa.func.from_unixtime(arg)
    elif unit_short == "us":
        res = sa.func.from_unixtime_nanos((arg - arg % 1_000_000) * 1_000)
    elif unit_short == "ns":
        res = sa.func.from_unixtime_nanos(arg - arg % 1_000_000_000)
    else:
        raise com.UnsupportedOperationError(f"{unit!r} unit is not supported")
    return sa.cast(res, t.get_sqla_type(op.dtype))


if_ = getattr(sa.func, "if")


def _neg_idx_to_pos(array, idx):
    arg_length = sa.func.cardinality(array)
    return if_(idx < 0, arg_length + sa.func.greatest(idx, -arg_length), idx)


def _array_slice(t, op):
    arg = t.translate(op.arg)

    arg_length = sa.func.cardinality(arg)

    if (start := op.start) is None:
        start = 0
    else:
        start = sa.func.least(arg_length, _neg_idx_to_pos(arg, t.translate(start)))

    if (stop := op.stop) is None:
        stop = arg_length
    else:
        stop = _neg_idx_to_pos(arg, t.translate(stop))

    length = stop - start
    return sa.func.slice(arg, start + 1, length, type_=arg.type)


def _extract_url_query(t, op):
    arg = t.translate(op.arg)
    key = op.key
    if key is None:
        result = sa.func.url_extract_query(arg)
    else:
        result = sa.func.url_extract_parameter(arg, t.translate(key))
    return sa.func.nullif(result, "")


def _round(t, op):
    arg = t.translate(op.arg)
    if (digits := op.digits) is not None:
        return sa.func.round(arg, t.translate(digits))
    return sa.func.round(arg)


def _unnest(t, op):
    arg = op.arg
    name = arg.name
    row_type = op.arg.dtype.value_type
    names = getattr(row_type, "names", (name,))
    rd = sa.func.unnest(t.translate(arg)).table_valued(*names).render_derived()
    # wrap in a row column so that we can return a single column from this rule
    if len(names) == 1:
        return rd.c[0]
    row = sa.func.row(*(rd.c[name] for name in names))
    return sa.cast(row, t.get_sqla_type(row_type))


def _where(t, op):
    return if_(
        t.translate(op.bool_expr),
        t.translate(op.true_expr),
        t.translate(op.false_null_expr),
        type_=t.get_sqla_type(op.dtype),
    )


def _cot(t, op):
    arg = t.translate(op.arg)
    return 1.0 / sa.func.tan(arg, type_=t.get_sqla_type(op.arg.dtype))


@compiles(array_map, "trino")
def compiles_list_apply(element, compiler, **kw):
    *args, signature, result = map(partial(compiler.process, **kw), element.clauses)
    return f"transform({', '.join(args)}, {signature} -> {result})"


def _array_map(t, op):
    return array_map(
        t.translate(op.arg),
        sa.literal_column(f"({op.parameter})"),
        t.translate(op.result),
    )


@compiles(array_filter, "trino")
def compiles_list_filter(element, compiler, **kw):
    *args, signature, result = map(partial(compiler.process, **kw), element.clauses)
    return f"filter({', '.join(args)}, {signature} -> {result})"


def _array_filter(t, op):
    return array_filter(
        t.translate(op.arg),
        sa.literal_column(f"({op.parameter})"),
        t.translate(op.result),
    )


def _first_last(t, op, *, offset: Literal[-1, 1]):
    return sa.func.element_at(t._reduction(sa.func.array_agg, op), offset)


def _zip(t, op):
    # more than one chunk means more than 5 arguments to zip, which trino
    # doesn't support
    #
    # help trino out by reducing in chunks of 5 using zip_with
    max_zip_arguments = 5
    chunks = (
        (len(chunk), sa.func.zip(*chunk) if len(chunk) > 1 else chunk[0])
        for chunk in toolz.partition_all(max_zip_arguments, map(t.translate, op.arg))
    )

    def combine_zipped(left, right):
        left_n, left_chunk = left
        lhs = (
            ", ".join(f"x[{i:d}]" for i in range(1, left_n + 1)) if left_n > 1 else "x"
        )

        right_n, right_chunk = right
        rhs = (
            ", ".join(f"y[{i:d}]" for i in range(1, right_n + 1))
            if right_n > 1
            else "y"
        )

        zipped_chunk = sa.func.zip_with(
            left_chunk, right_chunk, sa.literal_column(f"(x, y) -> ROW({lhs}, {rhs})")
        )
        return left_n + right_n, zipped_chunk

    all_n, chunk = reduce(combine_zipped, chunks)

    dtype = op.dtype

    assert all_n == len(dtype.value_type)

    return sa.type_coerce(chunk, t.get_sqla_type(dtype))


@compiles(try_cast, "trino")
def compiles_try_cast(element, compiler, **kw):
    return "TRY_CAST({} AS {})".format(
        compiler.process(element.clauses.clauses[0], **kw),
        compiler.visit_typeclause(element),
    )


def _try_cast(t, op):
    arg = t.translate(op.arg)
    to = t.get_sqla_type(op.to)
    return try_cast(arg, type_=to)


def _array_intersect(t, op):
    return t.translate(
        ops.ArrayFilter(op.left, func=lambda x: ops.ArrayContains(op.right, x))
    )


operation_registry.update(
    {
        # conditional expressions
        # static checks are not happy with using "if" as a property
        ops.Where: _where,
        # boolean reductions
        ops.Any: reduction(sa.func.bool_or),
        ops.All: reduction(sa.func.bool_and),
        ops.NotAny: reduction(lambda x: sa.func.bool_and(~x)),
        ops.NotAll: reduction(lambda x: sa.func.bool_or(~x)),
        ops.ArgMin: reduction(sa.func.min_by),
        ops.ArgMax: reduction(sa.func.max_by),
        # array ops
        ops.Correlation: _corr,
        ops.Covariance: _covar,
        ops.ExtractMillisecond: unary(sa.func.millisecond),
        ops.Arbitrary: _arbitrary,
        ops.ApproxCountDistinct: reduction(sa.func.approx_distinct),
        ops.ApproxMedian: reduction(lambda arg: sa.func.approx_percentile(arg, 0.5)),
        ops.RegexExtract: fixed_arity(sa.func.regexp_extract, 3),
        ops.RegexReplace: fixed_arity(sa.func.regexp_replace, 3),
        ops.RegexSearch: fixed_arity(
            lambda arg, pattern: sa.func.regexp_position(arg, pattern) != -1, 2
        ),
        ops.GroupConcat: _group_concat,
        ops.BitAnd: reduction(sa.func.bitwise_and_agg),
        ops.BitOr: reduction(sa.func.bitwise_or_agg),
        ops.BitXor: reduction(
            lambda arg: sa.func.reduce_agg(
                arg,
                0,
                sa.text("(a, b) -> bitwise_xor(a, b)"),
                sa.text("(a, b) -> bitwise_xor(a, b)"),
            )
        ),
        ops.BitwiseAnd: fixed_arity(sa.func.bitwise_and, 2),
        ops.BitwiseOr: fixed_arity(sa.func.bitwise_or, 2),
        ops.BitwiseXor: fixed_arity(sa.func.bitwise_xor, 2),
        ops.BitwiseLeftShift: fixed_arity(sa.func.bitwise_left_shift, 2),
        ops.BitwiseRightShift: fixed_arity(sa.func.bitwise_right_shift, 2),
        ops.BitwiseNot: unary(sa.func.bitwise_not),
        ops.ArrayCollect: reduction(sa.func.array_agg),
        ops.ArrayConcat: varargs(sa.func.concat),
        ops.ArrayLength: unary(sa.func.cardinality),
        ops.ArrayIndex: fixed_arity(
            lambda arg, index: sa.func.element_at(arg, index + 1), 2
        ),
        ops.ArrayColumn: _array_column,
        ops.ArrayRepeat: fixed_arity(
            lambda arg, times: sa.func.flatten(sa.func.repeat(arg, times)), 2
        ),
        ops.ArraySlice: _array_slice,
        ops.ArrayMap: _array_map,
        ops.ArrayFilter: _array_filter,
        ops.ArrayContains: fixed_arity(
            lambda arr, el: if_(
                arr != sa.null(),
                sa.func.coalesce(sa.func.contains(arr, el), sa.false()),
                sa.null(),
            ),
            2,
        ),
        ops.ArrayPosition: fixed_arity(
            lambda lst, el: sa.func.array_position(lst, el) - 1, 2
        ),
        ops.ArrayDistinct: fixed_arity(sa.func.array_distinct, 1),
        ops.ArraySort: fixed_arity(sa.func.array_sort, 1),
        ops.ArrayRemove: fixed_arity(sa.func.array_remove, 2),
        ops.ArrayUnion: fixed_arity(sa.func.array_union, 2),
        ops.JSONGetItem: _json_get_item,
        ops.ExtractDayOfYear: unary(sa.func.day_of_year),
        ops.ExtractWeekOfYear: unary(sa.func.week_of_year),
        ops.DayOfWeekIndex: unary(
            lambda arg: sa.cast(
                sa.cast(sa.func.day_of_week(arg) + 6, sa.SMALLINT) % 7, sa.SMALLINT
            )
        ),
        ops.DayOfWeekName: unary(lambda arg: sa.func.date_format(arg, "%W")),
        ops.ExtractEpochSeconds: unary(sa.func.to_unixtime),
        ops.Translate: fixed_arity(sa.func.translate, 3),
        ops.StrRight: fixed_arity(lambda arg, nchars: sa.func.substr(arg, -nchars), 2),
        ops.StringSplit: fixed_arity(sa.func.split, 2),
        ops.Repeat: fixed_arity(
            lambda value, count: sa.func.array_join(sa.func.repeat(value, count), ""), 2
        ),
        ops.DateTruncate: _timestamp_truncate,
        ops.TimestampTruncate: _timestamp_truncate,
        ops.DateFromYMD: fixed_arity(
            lambda y, m, d: sa.func.from_iso8601_date(
                sa.func.format("%04d-%02d-%02d", y, m, d)
            ),
            3,
        ),
        ops.TimeFromHMS: fixed_arity(
            lambda h, m, s: sa.cast(sa.func.format("%02d:%02d:%02d", h, m, s), sa.TIME),
            3,
        ),
        ops.TimestampFromYMDHMS: fixed_arity(
            lambda y, mo, d, h, m, s: sa.cast(
                sa.func.from_iso8601_timestamp(
                    sa.func.format("%04d-%02d-%02dT%02d:%02d:%02d", y, mo, d, h, m, s)
                ),
                sa.TIMESTAMP(timezone=False),
            ),
            6,
        ),
        ops.Strftime: fixed_arity(sa.func.date_format, 2),
        ops.StringToTimestamp: fixed_arity(sa.func.date_parse, 2),
        ops.TimestampNow: fixed_arity(sa.func.now, 0),
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.StructField: lambda t, op: t.translate(op.arg).op(".")(sa.text(op.field)),
        ops.StructColumn: lambda t, op: sa.cast(
            sa.func.row(*map(t.translate, op.values)), t.get_sqla_type(op.dtype)
        ),
        ops.Literal: _literal,
        ops.IfNull: fixed_arity(sa.func.coalesce, 2),
        ops.ZeroIfNull: unary(lambda value: sa.func.coalesce(value, 0)),
        ops.IsNan: unary(sa.func.is_nan),
        ops.IsInf: unary(sa.func.is_infinite),
        ops.Log: fixed_arity(lambda arg, base: sa.func.log(base, arg), 2),
        ops.Log2: unary(sa.func.log2),
        ops.Log10: unary(sa.func.log10),
        ops.MapLength: unary(sa.func.cardinality),
        ops.MapGet: fixed_arity(
            lambda arg, key, default: sa.func.coalesce(
                sa.func.element_at(arg, key), default
            ),
            3,
        ),
        ops.MapKeys: unary(sa.func.map_keys),
        ops.MapValues: unary(sa.func.map_values),
        ops.Map: fixed_arity(sa.func.map, 2),
        ops.MapMerge: fixed_arity(sa.func.map_concat, 2),
        ops.MapContains: fixed_arity(
            lambda arg, key: sa.func.contains(sa.func.map_keys(arg), key), 2
        ),
        ops.ExtractProtocol: unary(
            lambda arg: sa.func.nullif(sa.func.url_extract_protocol(arg), "")
        ),
        ops.ExtractHost: unary(
            lambda arg: sa.func.nullif(sa.func.url_extract_host(arg), "")
        ),
        ops.ExtractPath: unary(
            lambda arg: sa.func.nullif(sa.func.url_extract_path(arg), "")
        ),
        ops.ExtractFragment: unary(
            lambda arg: sa.func.nullif(sa.func.url_extract_fragment(arg), "")
        ),
        ops.ExtractFile: unary(
            lambda arg: sa.func.concat_ws(
                "?",
                sa.func.nullif(sa.func.url_extract_path(arg), ""),
                sa.func.nullif(sa.func.url_extract_query(arg), ""),
            )
        ),
        ops.ExtractQuery: _extract_url_query,
        ops.Cot: _cot,
        ops.Round: _round,
        ops.Pi: fixed_arity(sa.func.pi, 0),
        ops.E: fixed_arity(sa.func.e, 0),
        ops.Quantile: reduction(sa.func.approx_percentile),
        ops.MultiQuantile: reduction(sa.func.approx_percentile),
        ops.StringAscii: unary(
            lambda d: sa.func.codepoint(
                sa.func.cast(sa.func.substr(d, 1, 2), sa.VARCHAR(1))
            )
        ),
        ops.TypeOf: unary(sa.func.typeof),
        ops.Unnest: _unnest,
        ops.ArrayStringJoin: fixed_arity(
            lambda sep, arr: sa.func.array_join(arr, sep), 2
        ),
        ops.StartsWith: fixed_arity(sa.func.starts_with, 2),
        ops.Argument: lambda _, op: sa.literal_column(op.name),
        ops.First: partial(_first_last, offset=1),
        ops.Last: partial(_first_last, offset=-1),
        ops.ArrayZip: _zip,
        ops.TryCast: _try_cast,
        ops.ExtractMicrosecond: fixed_arity(
            # trino only seems to store milliseconds, but the result of
            # formatting always pads the right with 000
            lambda arg: sa.cast(sa.func.date_format(arg, "%f"), sa.INTEGER()),
            1,
        ),
        ops.Levenshtein: fixed_arity(sa.func.levenshtein_distance, 2),
        ops.ArrayIntersect: _array_intersect,
    }
)

_invalid_operations = {
    # ibis.expr.operations.analytic
    ops.CumulativeOp,
    ops.NTile,
    # ibis.expr.operations.reductions
    ops.MultiQuantile,
    ops.Quantile,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
