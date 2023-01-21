from __future__ import annotations

import itertools

import numpy as np
import sqlalchemy as sa

import ibis.expr.operations as ops
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


def _map(_, op):
    if not (
        isinstance(keys := op.keys, ops.Literal)
        and isinstance(values := op.values, ops.Literal)
    ):
        raise TypeError("Both keys and values of an `ibis.map` call must be literals")

    return sa.func.object_construct_keep_null(
        *itertools.chain.from_iterable(zip(keys.value, values.value))
    )


def _nth_value(t, op):
    if not isinstance(nth := op.nth, ops.Literal):
        raise TypeError(f"`nth` argument must be a literal Python int, got {type(nth)}")
    return sa.func.nth_value(t.translate(op.arg), nth.value + 1)


def _arbitrary(t, op):
    if op.how != "first":
        raise ValueError(
            "Snowflake only supports the `first` option for `.arbitrary()`"
        )

    # we can't use any_value here because it respects nulls
    #
    # yes it's slower, but it's also consistent with every other backend
    return t._reduction(sa.func.min, op)


_TIMESTAMP_UNITS_TO_SCALE = {"s": 0, "ms": 3, "us": 6, "ns": 9}

_SF_POS_INF = sa.func.to_double("Inf")
_SF_NEG_INF = sa.func.to_double("-Inf")
_SF_NAN = sa.func.to_double("NaN")

operation_registry.update(
    {
        ops.JSONGetItem: fixed_arity(sa.func.get, 2),
        ops.StringFind: _string_find,
        ops.MapKeys: unary(sa.func.object_keys),
        ops.MapGet: fixed_arity(
            lambda arg, key, default: sa.func.coalesce(
                sa.func.get(arg, key), sa.func.to_variant(default)
            ),
            3,
        ),
        ops.MapContains: fixed_arity(
            lambda arg, key: sa.func.array_contains(
                sa.func.to_variant(key), sa.func.object_keys(arg)
            ),
            2,
        ),
        ops.MapLength: unary(lambda arg: sa.func.array_size(sa.func.object_keys(arg))),
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
        # snowflake typeof only accepts VARIANT
        ops.ArrayIndex: fixed_arity(sa.func.get, 2),
        ops.ArrayLength: fixed_arity(sa.func.array_size, 1),
        ops.ArrayConcat: fixed_arity(sa.func.array_cat, 2),
        ops.ArrayColumn: lambda t, op: sa.func.array_construct(
            *map(t.translate, op.cols)
        ),
        ops.ArraySlice: _array_slice,
        ops.ArrayCollect: reduction(sa.func.array_agg),
        ops.StringSplit: fixed_arity(sa.func.split, 2),
        ops.Map: _map,
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
    }
)

_invalid_operations = {
    # ibis.expr.operations.analytic
    ops.CumulativeAll,
    ops.CumulativeAny,
    ops.CumulativeOp,
    ops.NTile,
    # ibis.expr.operations.array
    ops.ArrayRepeat,
    ops.Unnest,
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
