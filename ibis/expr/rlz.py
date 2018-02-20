import pytest
import enum
from toolz import curry
from ibis.compat import suppress
import ibis.common as com
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.expr.datatypes as dt

# TODO try to import cytoolz
from toolz import curry, compose, identity  # try to use cytoolz
from toolz import unique, curry


class validator(curry):
    pass


@validator
def oneof(inners, arg):
    """At least one of the inner valudators must pass"""
    for inner in inners:
        with suppress(com.IbisTypeError):
            return inner(arg)
    raise com.IbisTypeError('None of the {} are applicable on arg'.format(inners))


@validator
def allof(inners, arg):
    """All of the inner valudators must pass.

    The order of inner validators matters.

    Parameters
    ----------
    inners : List[validator]
      Functions are applied from right to left so allof([rule1, rule2], arg) is
      the same as rule1(rule2(arg)).
    arg : Any
      Value to be validated.

    Returns
    -------
    arg : Any
      Value maybe transformed by inner validators.
    """
    return compose(*inners)(arg)


@validator
def optional(inner, arg, default=None):
    if arg is None:
        if default is None:
            return None
        elif callable(default):
            arg = default()  # required by genname
        else:
            arg = default
    return inner(arg)


@validator
def isin(values, arg):
    if arg not in values:
        raise com.IbisTypeError('Value {!r} not in {!r}'.format(arg, values))
    if isinstance(values, dict):  # TODO check for mapping instead
        return values[arg]
    else:
        return arg


@validator
def memberof(obj, arg):
    if isinstance(arg, enum.Enum):
        enum.unique(obj)  # check that enum has unique values
        arg = arg.name

    if not hasattr(obj, arg):
        raise com.IbisTypeError('Value {!r} is not a member of '
                                '{!r}'.format(arg, obj))
    return getattr(obj, arg)


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


# TODO: change it to raise instead to locate all temporary noop validator
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


# TODO: previouse number rules allowed booleans by default
numeric = oneof([integer, floating, decimal])
temporal = oneof([timestamp, date, time])


# TODO: instead of inner might just
# allof(column, boolean)

@validator
def scalar(inner, arg):
    return instanceof(ir.ScalarExpr, inner(arg))


@validator
def column(inner, arg):
    return instanceof(ir.ColumnExpr, inner(arg))


table = instanceof(ir.TableExpr)
schema = instanceof(sch.Schema)
