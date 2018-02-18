import pytest

from toolz import curry
from ibis.compat import suppress
import ibis.common as com
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.expr.datatypes as dt
from ibis.expr.types import validator
from toolz import curry, compose, identity  # try to use cytoolz


# pass fail_message as default argument


@validator
def optional(inner, arg, default=None):
    if arg is None:
        if default is None:
            return None
        elif callable(default):
            arg = default()  # required by gennaame
        else:
            arg = default
    return inner(arg)


@validator
def isin(values, arg):
    if arg not in values:
        raise com.IbisTypeError('Value {!r} not in {!r}'.format(arg, values))
    if isinstance(values, dict):
        return values[arg]
    else:
        return arg


# @validator
# def string_options(options, arg, case_sensitive=True):
#     if not case_sensitive:
#         options = set(option.lower() for option in options)
#     return isin(options, arg)


@validator
def oneof(inners, arg):
    for inner in inners:
        with suppress(com.IbisTypeError):
            return inner(arg)
    raise com.IbisTypeError('None of the {} are applicable on arg'.format(inners))


@validator
def allof(inners, arg):
    return compose(*inners)(arg)


@validator
def listof(inner, arg, min_length=0):
    if not isinstance(arg, (tuple, list)):
        raise com.IbisTypeError('Arg is not an instance of list or tuple')
    if len(arg) < min_length:
        raise com.IbisTypeError(
            'Arg must have at least {} number of elements'.format(min_length)
        )
    return ir.sequence(list(map(inner, arg)))


@validator
def datatype(arg):
    return dt.dtype(arg)


@validator
def instanceof(klass, arg):
    """Require that a value has a particular Python type."""
    if not isinstance(arg, klass):
        raise com.IbisTypeError(
            '{!r} is not an instance of {!r}'.format(arg, klass)
        )
    return arg


@validator
def value(dtype, arg):
    if arg is None:
        raise com.IbisTypeError(
            'Value argument with datatype {} is mandatory'.format(dtype)
        )

    if not isinstance(arg, ir.Expr):
        arg = ir.literal(arg)

    if dt.issubtype(arg.type(), dtype):  # TODO: remove this, should not be required
        return arg  # subtype of expected
    if dt.castable(arg.type(), dtype):
        return arg  # implicitly castable
    else:
        raise com.IbisTypeError('Given argument with datatype {} is not subtype'
                                'of {} nor implicitly castable to it'.format(arg.type(), dtype))

noop = validator(identity)

any = value(dt.any)
#null = instanceof(dt.Null)#value(dt.null)
double = value(dt.double)
string = value(dt.string)
boolean = value(dt.boolean)
integer = value(dt.integer)
decimal = value(dt.decimal)
floating = value(dt.floating)
date = value(dt.date)
time = value(dt.time)
timestamp = value(dt.timestamp)


@validator
def interval(arg, units=None):  # TODO: pass all units by default
    arg = value(dt.interval, arg)
    unit = arg.type().unit
    if units is not None and unit not in units:
        msg = 'Interval unit `{}` is not among the allowed ones {}'
        raise com.IbisTypeError(msg.format(unit, units))
    return arg


numeric = oneof([integer, floating, decimal]) #  it should be numeric with allow_boolean
temporal = oneof([timestamp, date, time])

# column = instanceof(ir.ColumnExpr)
# scalar = instanceof(ir.ScalarExpr)  # consider using as optional


@validator
def scalar(inner, arg):
    return instanceof(ir.ScalarExpr, inner(arg))


@validator
def column(inner, arg):
    return instanceof(ir.ColumnExpr, inner(arg))



collection = instanceof((ir.ColumnExpr, ir.TableExpr))

schema = instanceof(sch.Schema)


# # TODO: insteaof allof
# chain = validator(compose)

# chain(column, integer)
