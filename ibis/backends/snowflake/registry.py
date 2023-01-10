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


def _random(t, op):
    min_value = sa.cast(0, sa.dialects.postgresql.FLOAT())
    max_value = sa.cast(1, sa.dialects.postgresql.FLOAT())
    return sa.func.uniform(min_value, max_value, sa.func.random())


def _day_of_week_name(t, op):
    return sa.case(
        value=sa.func.dayname(t.translate(op.arg)),
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
        ops.RandomScalar: _random,
        # time and dates
        ops.TimeFromHMS: fixed_arity(sa.func.time_from_parts, 3),
        # columns
        ops.DayOfWeekName: _day_of_week_name,
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
    # ibis.expr.operations.generic
    ops.TableArrayView,
    ops.TypeOf,
    # ibis.expr.operations.logical
    ops.ExistsSubquery,
    ops.NotExistsSubquery,
    # ibis.expr.operations.maps
    ops.MapKeys,
    # ibis.expr.operations.numeric
    ops.BitwiseAnd,
    ops.BitwiseLeftShift,
    ops.BitwiseNot,
    ops.BitwiseOr,
    ops.BitwiseRightShift,
    ops.BitwiseXor,
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
