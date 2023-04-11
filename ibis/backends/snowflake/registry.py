from __future__ import annotations

import itertools

import numpy as np
import sqlalchemy as sa
from snowflake.sqlalchemy import ARRAY, OBJECT, VARIANT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import sqltypes
from sqlalchemy.sql.elements import Cast
from sqlalchemy.sql.functions import GenericFunction

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    geospatial_functions,
    reduction,
    unary,
)
from ibis.backends.postgres.registry import _literal as _postgres_literal
from ibis.backends.postgres.registry import operation_registry as _operation_registry

operation_registry = {
    op: _operation_registry[op]
    for op in _operation_registry.keys() - geospatial_functions.keys()
}


def _literal(t, op):
    value = op.value
    dtype = op.output_dtype

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
        if (tz := value.tzinfo) is not None:
            return sa.func.timestamp_tz_from_parts(*args, str(tz))
        else:
            return sa.func.timestamp_from_parts(*args)
    elif dtype.is_date():
        return sa.func.date_from_parts(value.year, value.month, value.day)
    elif dtype.is_array():
        return sa.func.array_construct(*value)
    elif dtype.is_map() or dtype.is_struct():
        return sa.func.object_construct_keep_null(
            *itertools.chain.from_iterable(value.items())
        )
    elif dtype.is_uuid():
        return sa.literal(str(value))
    return _postgres_literal(t, op)


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
        r = sa.func.get(sa.func.get(parsed_url, 'parameters'), t.translate(key))
    else:
        r = sa.func.get(parsed_url, 'query')

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
    if op.how != "first":
        raise com.UnsupportedOperationError(
            "Snowflake only supports the `first` option for `.arbitrary()`"
        )

    # we can't use any_value here because it respects nulls
    #
    # yes it's slower, but it's also consistent with every other backend
    return t._reduction(sa.func.min, op)


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


class _flatten(GenericFunction):
    def __init__(self, arg, *, type: sa.types.TypeEngine) -> None:
        super().__init__(arg)
        self.type = sqltypes.TableValueType(sa.Column("value", type))


@compiles(_flatten, "snowflake")
def compiles_flatten(element, compiler, **kw):
    arg = compiler.function_argspec(element, **kw)
    return f"FLATTEN(INPUT => {arg}, MODE => 'ARRAY')"


def _unnest(t, op):
    arg = t.translate(op.arg)
    # HACK: https://community.snowflake.com/s/question/0D50Z000086MVhnSAG/has-anyone-found-a-way-to-unnest-an-array-without-loosing-the-null-values
    sep = util.guid()
    sqla_type = t.get_sqla_type(op.output_dtype)
    col = (
        _flatten(sa.func.split(sa.func.array_to_string(arg, sep), sep), type=sqla_type)
        .lateral()
        .c["value"]
    )
    return sa.cast(sa.func.nullif(col, ""), type_=sqla_type)


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


_TIMESTAMP_UNITS_TO_SCALE = {"s": 0, "ms": 3, "us": 6, "ns": 9}

_SF_POS_INF = sa.func.to_double("Inf")
_SF_NEG_INF = sa.func.to_double("-Inf")
_SF_NAN = sa.func.to_double("NaN")

operation_registry.update(
    {
        ops.JSONGetItem: fixed_arity(sa.func.get, 2),
        ops.StringFind: _string_find,
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
        ops.MapGet: fixed_arity(
            lambda arg, key, default: sa.func.coalesce(
                sa.func.get(arg, key), sa.func.to_variant(default)
            ),
            3,
        ),
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
        ops.Where: fixed_arity(sa.func.iff, 3),
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
        ops.ArrayConcat: fixed_arity(sa.func.array_cat, 2),
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
            # snowflake is zero-based here, so we don't need to substract 1 from the result
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
        ops.StringSplit: fixed_arity(sa.func.split, 2),
        # snowflake typeof only accepts VARIANT, so we cast
        ops.TypeOf: unary(lambda arg: sa.func.typeof(sa.func.to_variant(arg))),
        ops.All: reduction(sa.func.booland_agg),
        ops.NotAll: reduction(lambda arg: ~sa.func.booland_agg(arg)),
        ops.Any: reduction(sa.func.boolor_agg),
        ops.NotAny: reduction(lambda arg: ~sa.func.boolor_agg(arg)),
        ops.BitAnd: reduction(sa.func.bitand_agg),
        ops.BitOr: reduction(sa.func.bitor_agg),
        ops.BitXor: reduction(sa.func.bitxor_agg),
        ops.DateFromYMD: fixed_arity(sa.func.date_from_parts, 3),
        ops.StringToTimestamp: fixed_arity(sa.func.to_timestamp_tz, 2),
        ops.RegexExtract: fixed_arity(sa.func.regexp_substr, 3),
        ops.RegexSearch: fixed_arity(sa.sql.operators.custom_op("REGEXP"), 2),
        ops.RegexReplace: fixed_arity(sa.func.regexp_replace, 3),
        ops.ExtractMillisecond: fixed_arity(
            lambda arg: sa.cast(
                sa.extract("epoch_millisecond", arg) % 1000, sa.SMALLINT
            ),
            1,
        ),
        ops.TimestampFromYMDHMS: fixed_arity(sa.func.timestamp_from_parts, 6),
        ops.TimestampFromUNIX: lambda t, op: sa.func.to_timestamp(
            t.translate(op.arg), _TIMESTAMP_UNITS_TO_SCALE[op.unit]
        ),
        ops.StructField: lambda t, op: sa.cast(
            sa.func.get(t.translate(op.arg), op.field), t.get_sqla_type(op.output_dtype)
        ),
        ops.NthValue: _nth_value,
        ops.Arbitrary: _arbitrary,
        ops.StructColumn: lambda t, op: sa.func.object_construct_keep_null(
            *itertools.chain.from_iterable(zip(op.names, map(t.translate, op.values)))
        ),
        ops.Unnest: _unnest,
        ops.ArgMin: reduction(sa.func.min_by),
        ops.ArgMax: reduction(sa.func.max_by),
        ops.ToJSONArray: lambda t, op: t.translate(ops.Cast(op.arg, op.output_dtype)),
        ops.ToJSONMap: lambda t, op: t.translate(ops.Cast(op.arg, op.output_dtype)),
        ops.StartsWith: fixed_arity(sa.func.startswith, 2),
        ops.EndsWith: fixed_arity(sa.func.endswith, 2),
        ops.GroupConcat: _group_concat,
    }
)

_invalid_operations = {
    # ibis.expr.operations.analytic
    ops.CumulativeAll,
    ops.CumulativeAny,
    ops.CumulativeOp,
    ops.NTile,
    # ibis.expr.operations.array
    ops.ArrayRemove,
    ops.ArrayRepeat,
    ops.ArraySort,
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
