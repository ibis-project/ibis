from __future__ import annotations

import numpy as np
import sqlalchemy as sa
from snowflake.sqlalchemy.custom_types import VARIANT

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
        value=sa.func.dayname(arg),
        whens=[
            ("Sun", "Sunday"),
            ("Mon", "Monday"),
            ("Tue", "Tuesday"),
            ("Wed", "Wednesday"),
            ("Thu", "Thursday"),
            ("Fri", "Friday"),
            ("Sat", "Saturday"),
        ],
        else_=None,
    )


def _extract_url_query(t, op):
    parsed_url = sa.func.parse_url(t.translate(op.arg), 1)

    if (key := op.key) is not None:
        r = sa.func.get(sa.func.get(parsed_url, 'parameters'), t.translate(key))
    else:
        r = sa.func.get(parsed_url, 'query')

    return sa.func.nullif(sa.func.as_varchar(r), "")


_SF_POS_INF = sa.cast(sa.literal("Inf"), sa.FLOAT)
_SF_NEG_INF = -_SF_POS_INF
_SF_NAN = sa.cast(sa.literal("NaN"), sa.FLOAT)

operation_registry.update(
    {
        ops.JSONGetItem: fixed_arity(sa.func.get, 2),
        ops.StructField: fixed_arity(sa.func.get, 2),
        ops.StringFind: _string_find,
        ops.MapKeys: fixed_arity(sa.func.object_keys, 1),
        ops.BitwiseLeftShift: fixed_arity(sa.func.bitshiftleft, 2),
        ops.BitwiseRightShift: fixed_arity(sa.func.bitshiftright, 2),
        ops.Ln: fixed_arity(sa.func.ln, 1),
        ops.Log2: fixed_arity(lambda arg: sa.func.log(2, arg), 1),
        ops.Log10: fixed_arity(lambda arg: sa.func.log(10, arg), 1),
        ops.Log: fixed_arity(lambda arg, base: sa.func.log(base, arg), 2),
        ops.IsInf: fixed_arity(lambda arg: arg.in_((_SF_POS_INF, _SF_NEG_INF)), 1),
        ops.IsNan: fixed_arity(lambda arg: arg == _SF_NAN, 1),
        ops.Literal: _literal,
        ops.Round: _round,
        ops.Modulus: fixed_arity(sa.func.mod, 2),
        ops.Mode: reduction(sa.func.mode),
        ops.Where: fixed_arity(sa.func.iff, 3),
        # numbers
        ops.RandomScalar: fixed_arity(
            lambda: sa.func.uniform(
                sa.cast(0, sa.dialects.postgresql.FLOAT()),
                sa.cast(1, sa.dialects.postgresql.FLOAT()),
                sa.func.random(),
            ),
            0,
        ),
        # time and dates
        ops.TimeFromHMS: fixed_arity(sa.func.time_from_parts, 3),
        # columns
        ops.DayOfWeekName: fixed_arity(_day_of_week_name, 1),
        ops.ExtractProtocol: fixed_arity(
            lambda arg: sa.func.nullif(
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "scheme")), ""
            ),
            1,
        ),
        ops.ExtractAuthority: fixed_arity(
            lambda arg: sa.func.concat_ws(
                ":",
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "host")),
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "port")),
            ),
            1,
        ),
        ops.ExtractFile: fixed_arity(
            lambda arg: sa.func.concat_ws(
                "?",
                "/"
                + sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "path")),
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "query")),
            ),
            1,
        ),
        ops.ExtractPath: fixed_arity(
            lambda arg: (
                "/" + sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "path"))
            ),
            1,
        ),
        ops.ExtractQuery: _extract_url_query,
        ops.ExtractFragment: fixed_arity(
            lambda arg: sa.func.nullif(
                sa.func.as_varchar(sa.func.get(sa.func.parse_url(arg, 1), "fragment")),
                "",
            ),
            1,
        ),
        # snowflake typeof only accepts VARIANT
        ops.TypeOf: unary(lambda arg: sa.func.typeof(sa.cast(arg, VARIANT))),
    }
)

_invalid_operations = {
    # ibis.expr.operations.analytic
    ops.CumulativeAll,
    ops.CumulativeAny,
    ops.CumulativeOp,
    ops.NTile,
    ops.NthValue,
    # ibis.expr.operations.array
    ops.ArrayColumn,
    ops.ArrayConcat,
    ops.ArrayIndex,
    ops.ArrayLength,
    ops.ArrayRepeat,
    ops.ArraySlice,
    ops.Unnest,
    # ibis.expr.operations.maps
    ops.MapKeys,
    # ibis.expr.operations.reductions
    ops.All,
    ops.Any,
    ops.ArrayCollect,
    ops.BitAnd,
    ops.BitOr,
    ops.BitXor,
    ops.MultiQuantile,
    ops.NotAll,
    ops.NotAny,
    # ibis.expr.operations.strings
    ops.FindInSet,
    ops.RegexExtract,
    ops.RegexReplace,
    ops.RegexSearch,
    ops.StringSplit,
    # ibis.expr.operations.structs
    ops.StructField,
    # ibis.expr.operations.temporal
    ops.DateFromYMD,
    ops.ExtractMillisecond,
    ops.IntervalFromInteger,
    ops.StringToTimestamp,
    ops.TimestampDiff,
    ops.TimestampFromUNIX,
    ops.TimestampFromYMDHMS,
}

operation_registry = {
    k: v for k, v in operation_registry.items() if k not in _invalid_operations
}
