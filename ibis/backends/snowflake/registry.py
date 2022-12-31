import numpy as np
import sqlalchemy as sa

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    geospatial_functions,
    reduction,
)
from ibis.backends.postgres.registry import _literal as _postgres_literal
from ibis.backends.postgres.registry import operation_registry as _operation_registry

operation_registry = {
    op: _operation_registry[op]
    for op in _operation_registry.keys() - geospatial_functions.keys()
}


def _literal(t, op):
    if isinstance(op, ops.Literal) and op.output_dtype.is_floating():
        value = op.value

        if np.isnan(value):
            return _SF_NAN

        if np.isinf(value):
            return _SF_NEG_INF if value < 0 else _SF_POS_INF
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


_PARSE_URL_FUNCS = {
    "PROTOCOL": lambda url, _: sa.func.as_varchar(sa.func.get(url, "scheme")),
    "PATH": lambda url, _: "/" + sa.func.as_varchar(sa.func.get(url, "path")),
    "REF": lambda url, _: sa.func.as_varchar(sa.func.get(url, "fragment")),
    "AUTHORITY": lambda url, _: sa.func.concat_ws(
        ":",
        sa.func.as_varchar(sa.func.get(url, "host")),
        sa.func.as_varchar(sa.func.get(url, "port")),
    ),
    "FILE": lambda url, _: sa.func.concat_ws(
        "?",
        "/" + sa.func.as_varchar(sa.func.get(url, "path")),
        sa.func.as_varchar(sa.func.get(url, "query")),
    ),
    "QUERY": lambda url, key: sa.func.as_varchar(
        sa.func.get(sa.func.get(url, "query"), key)
        if key is not None
        else sa.func.get(url, "query")
    ),
}


def _parse_url(t, op):
    if (func := _PARSE_URL_FUNCS.get(extract := op.extract)) is None:
        raise ValueError(f"`{extract}` is not supported in the Snowflake backend")

    return sa.func.nullif(
        func(
            sa.func.parse_url(t.translate(op.arg), 1),
            t.translate(key) if (key := op.key) is not None else key,
        ),
        "",
    )


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
        ops.ParseURL: _parse_url,
    }
)
