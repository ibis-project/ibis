from __future__ import annotations

import functools
import itertools

import numpy as np
import sqlalchemy as sa
from snowflake.sqlalchemy import ARRAY, OBJECT, VARIANT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import Cast

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    geospatial_functions,
    get_col,
    get_sqla_table,
    reduction,
    unary,
    varargs,
)
from ibis.backends.postgres.registry import _literal as _postgres_literal
from ibis.backends.postgres.registry import operation_registry as _operation_registry

operation_registry = {
    op: _operation_registry[op]
    for op in _operation_registry.keys() - geospatial_functions.keys()
}


def _literal(t, op):
    value = op.value
    dtype = op.dtype

    if value is None:
        return sa.null()

    if dtype.is_floating():
        if np.isnan(value):
            return _SF_NAN

        if np.isinf(value):
            return _SF_NEG_INF if value < 0 else _SF_POS_INF
    elif dtype.is_timestamp():
        args = (
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond * 1_000,
        )
        if value.tzinfo is not None:
            return sa.func.timestamp_tz_from_parts(*args, dtype.timezone)
        else:
            return sa.func.timestamp_from_parts(*args)
    elif dtype.is_date():
        return sa.func.date_from_parts(value.year, value.month, value.day)
    elif dtype.is_time():
        nanos = value.microsecond * 1_000
        return sa.func.time_from_parts(value.hour, value.minute, value.second, nanos)
    elif dtype.is_array():
        return sa.func.array_construct(*value)
    elif dtype.is_map() or dtype.is_struct():
        return sa.func.object_construct_keep_null(
            *itertools.chain.from_iterable(value.items())
        )
    elif dtype.is_uuid():
        return sa.literal(str(value))
    return _postgres_literal(t, op)


def _table_column(t, op):
    ctx = t.context
    table = op.table

    sa_table = get_sqla_table(ctx, table)
    out_expr = get_col(sa_table, op)

    if (dtype := op.dtype).is_timestamp() and (timezone := dtype.timezone) is not None:
        out_expr = sa.func.convert_timezone(timezone, out_expr).label(op.name)

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if t.permit_subquery and ctx.is_foreign_expr(table):
        return sa.select(out_expr)

    return out_expr


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


def _day_of_week_name(arg):
    return sa.case(
        ("Sun", "Sunday"),
        ("Mon", "Monday"),
        ("Tue", "Tuesday"),
        ("Wed", "Wednesday"),
        ("Thu", "Thursday"),
        ("Fri", "Friday"),
        ("Sat", "Saturday"),
        value=sa.func.dayname(arg),
        else_=None,
    )


def _extract_url_query(t, op):
    parsed_url = sa.func.parse_url(t.translate(op.arg), 1)

    if (key := op.key) is not None:
        r = sa.func.get(sa.func.get(parsed_url, "parameters"), t.translate(key))
    else:
        r = sa.func.get(parsed_url, "query")

    return sa.func.nullif(sa.func.as_varchar(r), "")


def _array_slice(t, op):
    arg = t.translate(op.arg)

    if (start := op.start) is not None:
        start = t.translate(start)
    else:
        start = 0

    if (stop := op.stop) is not None:
        stop = t.translate(stop)
    else:
        stop = sa.func.array_size(arg)

    return sa.func.array_slice(t.translate(op.arg), start, stop)


def _nth_value(t, op):
    if not isinstance(nth := op.nth, ops.Literal):
        raise TypeError(f"`nth` argument must be a literal Python int, got {type(nth)}")
    return sa.func.nth_value(t.translate(op.arg), nth.value + 1)


def _arbitrary(t, op):
    if (how := op.how) == "first":
        return t._reduction(lambda x: sa.func.get(sa.func.array_agg(x), 0), op)
    elif how == "last":
        return t._reduction(
            lambda x: sa.func.get(
                sa.func.array_agg(x), sa.func.array_size(sa.func.array_agg(x)) - 1
            ),
            op,
        )
    else:
        raise com.UnsupportedOperationError("how must be 'first' or 'last'")


@compiles(Cast, "snowflake")
def compiles_cast(element, compiler, **kw):
    typ = compiler.visit_typeclause(element, **kw)
    if typ in ("OBJECT", "ARRAY"):
        arg = compiler.process(element.clause, **kw)
        return f"IFF(IS_{typ}({arg}), {arg}, NULL)"
    return compiler.visit_cast(element, **kw)


@compiles(sa.TEXT, "snowflake")
@compiles(sa.VARCHAR, "snowflake")
def compiles_string(element, compiler, **kw):
    return "VARCHAR"


@compiles(OBJECT, "snowflake")
@compiles(ARRAY, "snowflake")
@compiles(VARIANT, "snowflake")
def compiles_object_type(element, compiler, **kw):
    return type(element).__name__.upper()


def _unnest(t, op):
    arg = t.translate(op.arg)
    # HACK: https://community.snowflake.com/s/question/0D50Z000086MVhnSAG/has-anyone-found-a-way-to-unnest-an-array-without-loosing-the-null-values
    sep = util.guid()
    col = sa.func.nullif(
        sa.func.split_to_table(sa.func.array_to_string(arg, sep), sep)
        .table_valued("value")  # seq, index, value is supported but we only need value
        .lateral()
        .c["value"],
        "",
    )
    return sa.cast(
        sa.func.coalesce(sa.func.try_parse_json(col), sa.func.to_variant(col)),
        type_=t.get_sqla_type(op.dtype),
    )


def _group_concat(t, op):
    if (where := op.where) is None:
        return sa.func.listagg(t.translate(op.arg), t.translate(op.sep))

    where_sa = t.translate(where)
    arg_sa = sa.func.iff(where_sa, t.translate(op.arg), None)

    return sa.func.iff(
        sa.func.count_if(arg_sa != sa.null()) != 0,
        sa.func.listagg(arg_sa, t.translate(op.sep)),
        None,
    )


def _array_zip(t, op):
    return sa.type_coerce(
        sa.func.ibis_udfs.public.array_zip(
            sa.func.array_construct(*map(t.translate, op.arg))
        ),
        t.get_sqla_type(op.dtype),
    )


def _regex_extract(t, op):
    arg = t.translate(op.arg)
    pattern = t.translate(op.pattern)
    index = t.translate(op.index)
    # https://docs.snowflake.com/en/sql-reference/functions/regexp_substr
    return sa.func.regexp_substr(arg, pattern, 1, 1, "ce", index)


def _map_get(t, op):
    arg = op.arg
    key = op.key
    default = op.default
    dtype = op.dtype
    sqla_type = t.get_sqla_type(dtype)
    expr = sa.func.coalesce(
        sa.func.get(t.translate(arg), t.translate(key)),
        sa.func.to_variant(t.translate(default)),
        type_=sqla_type,
    )
    if dtype.is_json() or dtype.is_null():
        return expr

    # cast if ibis thinks the value type is not JSON
    #
    # this ensures that we can get deserialized map values even though maps are
    # always JSON in the value type inside snowflake
    return sa.cast(expr, sqla_type)


def _timestamp_bucket(t, op):
    if op.offset is not None:
        raise com.UnsupportedOperationError(
            "`offset` is not supported in the Snowflake backend for timestamp bucketing"
        )

    interval = op.interval

    if not isinstance(interval, ops.Literal):
        raise com.UnsupportedOperationError(
            f"Interval must be a literal for the Snowflake backend, got {type(interval)}"
        )

    return sa.func.time_slice(
        t.translate(op.arg), interval.value, interval.dtype.unit.name
    )


_TIMESTAMP_UNITS_TO_SCALE = {"s": 0, "ms": 3, "us": 6, "ns": 9}

_SF_POS_INF = sa.func.to_double("Inf")
_SF_NEG_INF = sa.func.to_double("-Inf")
_SF_NAN = sa.func.to_double("NaN")

operation_registry.update(
    {
        ops.JSONGetItem: fixed_arity(sa.func.get, 2),
        ops.StringFind: _string_find,
        ops.ArrayZip: _array_zip,
        ops.Map: fixed_arity(
            lambda keys, values: sa.func.iff(
                sa.func.is_array(keys) & sa.func.is_array(values),
                sa.func.ibis_udfs.public.object_from_arrays(keys, values),
                sa.null(),
            ),
            2,
        ),
        ops.MapKeys: unary(
            lambda arg: sa.func.iff(
                sa.func.is_object(arg), sa.func.object_keys(arg), sa.null()
            )
        ),
        ops.MapValues: unary(
            lambda arg: sa.func.iff(
                sa.func.is_object(arg),
                sa.func.ibis_udfs.public.object_values(arg),
                sa.null(),
            )
        ),
        ops.MapGet: _map_get,
        ops.MapContains: fixed_arity(
            lambda arg, key: sa.func.array_contains(
                sa.func.to_variant(key),
                sa.func.iff(
                    sa.func.is_object(arg), sa.func.object_keys(arg), sa.null()
                ),
            ),
            2,
        ),
        ops.MapMerge: fixed_arity(
            lambda a, b: sa.func.iff(
                sa.func.is_object(a) & sa.func.is_object(b),
                sa.func.ibis_udfs.public.object_merge(a, b),
                sa.null(),
            ),
            2,
        ),
        ops.MapLength: unary(
            lambda arg: sa.func.array_size(
                sa.func.iff(sa.func.is_object(arg), sa.func.object_keys(arg), sa.null())
            )
        ),
        ops.BitwiseAnd: fixed_arity(sa.func.bitand, 2),
        ops.BitwiseNot: unary(sa.func.bitnot),
        ops.BitwiseOr: fixed_arity(sa.func.bitor, 2),
        ops.BitwiseXor: fixed_arity(sa.func.bitxor, 2),
        ops.BitwiseLeftShift: fixed_arity(sa.func.bitshiftleft, 2),
        ops.BitwiseRightShift: fixed_arity(sa.func.bitshiftright, 2),
        ops.Ln: unary(sa.func.ln),
        ops.Log2: unary(lambda arg: sa.func.log(2, arg)),
        ops.Log10: unary(lambda arg: sa.func.log(10, arg)),
        ops.Log: fixed_arity(lambda arg, base: sa.func.log(base, arg), 2),
        ops.IsInf: unary(lambda arg: arg.in_((_SF_POS_INF, _SF_NEG_INF))),
        ops.IsNan: unary(lambda arg: arg == _SF_NAN),
        ops.Literal: _literal,
        ops.Round: _round,
        ops.Modulus: fixed_arity(sa.func.mod, 2),
        ops.Mode: reduction(sa.func.mode),
        ops.IfElse: fixed_arity(sa.func.iff, 3),
        # numbers
        ops.RandomScalar: fixed_arity(
            lambda: sa.func.uniform(
                sa.func.to_double(0.0), sa.func.to_double(1.0), sa.func.random()
            ),
            0,
        ),
        # time and dates
        ops.TimeFromHMS: fixed_arity(sa.func.time_from_parts, 3),
        # columns
        ops.DayOfWeekName: unary(_day_of_week_name),
        ops.ExtractProtocol: unary(
            lambda arg: sa.func.nullif(
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "scheme")), ""
            )
        ),
        ops.ExtractAuthority: unary(
            lambda arg: sa.func.concat_ws(
                ":",
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "host")),
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "port")),
            )
        ),
        ops.ExtractFile: unary(
            lambda arg: sa.func.concat_ws(
                "?",
                "/"
                + sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "path")),
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "query")),
            )
        ),
        ops.ExtractPath: unary(
            lambda arg: (
                "/" + sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "path"))
            )
        ),
        ops.ExtractQuery: _extract_url_query,
        ops.ExtractFragment: unary(
            lambda arg: sa.func.nullif(
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "fragment")),
                "",
            )
        ),
        ops.ArrayIndex: fixed_arity(sa.func.get, 2),
        ops.ArrayLength: fixed_arity(sa.func.array_size, 1),
        ops.ArrayConcat: varargs(
            lambda *args: functools.reduce(sa.func.array_cat, args)
        ),
        ops.ArrayColumn: lambda t, op: sa.func.array_construct(
            *map(t.translate, op.cols)
        ),
        ops.ArraySlice: _array_slice,
        ops.ArrayCollect: reduction(
            lambda arg: sa.func.array_agg(
                sa.func.ifnull(arg, sa.func.parse_json("null")), type_=ARRAY
            )
        ),
        ops.ArrayContains: fixed_arity(
            lambda arr, el: sa.func.array_contains(sa.func.to_variant(el), arr), 2
        ),
        ops.ArrayPosition: fixed_arity(
            # snowflake is zero-based here, so we don't need to subtract 1 from the result
            lambda lst, el: sa.func.coalesce(
                sa.func.array_position(sa.func.to_variant(el), lst), -1
            ),
            2,
        ),
        ops.ArrayDistinct: fixed_arity(sa.func.array_distinct, 1),
        ops.ArrayUnion: fixed_arity(
            lambda left, right: sa.func.array_distinct(sa.func.array_cat(left, right)),
            2,
        ),
        ops.ArrayRemove: fixed_arity(sa.func.array_remove, 2),
        ops.ArrayIntersect: fixed_arity(sa.func.array_intersection, 2),
        ops.StringSplit: fixed_arity(sa.func.split, 2),
        # snowflake typeof only accepts VARIANT, so we cast
        ops.TypeOf: unary(lambda arg: sa.func.typeof(sa.func.to_variant(arg))),
        ops.All: reduction(sa.func.booland_agg),
        ops.Any: reduction(sa.func.boolor_agg),
        ops.BitAnd: reduction(sa.func.bitand_agg),
        ops.BitOr: reduction(sa.func.bitor_agg),
        ops.BitXor: reduction(sa.func.bitxor_agg),
        ops.DateFromYMD: fixed_arity(sa.func.date_from_parts, 3),
        ops.StringToTimestamp: fixed_arity(sa.func.to_timestamp_tz, 2),
        ops.RegexExtract: _regex_extract,
        ops.RegexSearch: fixed_arity(
            lambda arg, pattern: sa.func.regexp_instr(arg, pattern) != 0, 2
        ),
        ops.RegexReplace: fixed_arity(sa.func.regexp_replace, 3),
        ops.ExtractMicrosecond: fixed_arity(
            lambda arg: sa.cast(
                sa.extract("epoch_microsecond", arg) % 1000000, sa.SMALLINT
            ),
            1,
        ),
        ops.ExtractMillisecond: fixed_arity(
            lambda arg: sa.cast(
                sa.extract("epoch_millisecond", arg) % 1000, sa.SMALLINT
            ),
            1,
        ),
        ops.TimestampFromYMDHMS: fixed_arity(sa.func.timestamp_from_parts, 6),
        ops.TimestampFromUNIX: lambda t, op: sa.func.to_timestamp(
            t.translate(op.arg), _TIMESTAMP_UNITS_TO_SCALE[op.unit.short]
        ),
        ops.StructField: lambda t, op: sa.cast(
            sa.func.get(t.translate(op.arg), op.field), t.get_sqla_type(op.dtype)
        ),
        ops.NthValue: _nth_value,
        ops.Arbitrary: _arbitrary,
        ops.First: reduction(lambda x: sa.func.get(sa.func.array_agg(x), 0)),
        ops.Last: reduction(
            lambda x: sa.func.get(
                sa.func.array_agg(x), sa.func.array_size(sa.func.array_agg(x)) - 1
            )
        ),
        ops.StructColumn: lambda t, op: sa.func.object_construct_keep_null(
            *itertools.chain.from_iterable(zip(op.names, map(t.translate, op.values)))
        ),
        ops.Unnest: _unnest,
        ops.ArgMin: reduction(sa.func.min_by),
        ops.ArgMax: reduction(sa.func.max_by),
        ops.ToJSONArray: lambda t, op: t.translate(ops.Cast(op.arg, op.dtype)),
        ops.ToJSONMap: lambda t, op: t.translate(ops.Cast(op.arg, op.dtype)),
        ops.StartsWith: fixed_arity(sa.func.startswith, 2),
        ops.EndsWith: fixed_arity(sa.func.endswith, 2),
        ops.GroupConcat: _group_concat,
        ops.Hash: unary(sa.func.hash),
        ops.ApproxMedian: reduction(lambda x: sa.func.approx_percentile(x, 0.5)),
        ops.Median: reduction(sa.func.median),
        ops.TableColumn: _table_column,
        ops.Levenshtein: fixed_arity(sa.func.editdistance, 2),
        ops.ArraySort: unary(sa.func.ibis_udfs.public.array_sort),
        ops.ArrayRepeat: fixed_arity(sa.func.ibis_udfs.public.array_repeat, 2),
        ops.TimeDelta: fixed_arity(
            lambda part, left, right: sa.func.timediff(part, right, left), 3
        ),
        ops.DateDelta: fixed_arity(
            lambda part, left, right: sa.func.datediff(part, right, left), 3
        ),
        ops.TimestampDelta: fixed_arity(
            lambda part, left, right: sa.func.timestampdiff(part, right, left), 3
        ),
        ops.TimestampBucket: _timestamp_bucket,
    }
)

_invalid_operations = {
    # ibis.expr.operations.array
    ops.ArrayMap,
    ops.ArrayFilter,
    # ibis.expr.operations.reductions
    ops.MultiQuantile,
    # ibis.expr.operations.strings
    ops.FindInSet,
    # ibis.expr.operations.temporal
    ops.IntervalFromInteger,
    ops.TimestampDiff,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
