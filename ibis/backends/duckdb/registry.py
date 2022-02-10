import sqlalchemy as sa

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import unary

from ..postgres.registry import fixed_arity, operation_registry

operation_registry = operation_registry.copy()


def _round(t, expr):
    arg, digits = expr.op().args
    sa_arg = t.translate(arg)

    if digits is None:
        return sa.func.round(sa_arg)

    result = sa.func.round(sa_arg, t.translate(digits))
    return result


def _mod(t, expr):
    left, right = map(t.translate, expr.op().args)
    return left % right


def _log(t, expr):
    arg, base = expr.op().args
    sa_arg = t.translate(arg)
    if base is not None:
        sa_base = t.translate(base)
        if sa_base.value == 2:
            return sa.func.log2(sa_arg)
        elif sa_base.value == 10:
            return sa.func.log(sa_arg)
        else:
            raise NotImplementedError
        return sa.func.log(sa_base, sa_arg)
    return sa.func.ln(sa_arg)


def _timestamp_from_unix(t, expr):
    op = expr.op()
    arg, unit = op.args
    arg = t.translate(arg)

    if unit in {"us", "ns"}:
        raise ValueError(f"`{unit}` unit is not supported!")

    if unit == "ms":
        return sa.func.epoch_ms(arg)
    elif unit == "s":
        return sa.func.to_timestamp(arg)


# TODO(gil): this is working except the results of the
# substraction are being truncated
def _timestamp_diff(t, expr):
    sa_left, sa_right = map(t.translate, expr.op().args)
    ts = sa.text(f"TIMESTAMP '{sa_right.value}'")
    return sa_left.op("-")(ts)


operation_registry.update(
    {
        ops.Round: _round,
        ops.Log2: unary(sa.func.log2),
        ops.Modulus: _mod,
        ops.Log: _log,
        ops.Translate: fixed_arity('replace', 3),
        ops.DayOfWeekName: unary(sa.func.dayname),
        ops.TimestampFromUNIX: _timestamp_from_unix,
        # TODO(gil): this should work except sqlalchemy doesn't know how to
        # render duckdb timestamps
        # ops.TimestampDiff: fixed_arity('age', 2),
        ops.TimestampDiff: _timestamp_diff,
    }
)
