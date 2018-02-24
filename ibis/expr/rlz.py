import pytest
import enum
from toolz import curry
from ibis.compat import suppress
import ibis.util as util
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


noop = validator(identity)


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
        raise ValueError('Value {!r} not in {!r}'.format(arg, values))
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
    # TODO support DataType classes, not just instances
    if arg is None:
        raise com.IbisTypeError('Passing value argument with datatype {} is '
                                'mandatory'.format(dtype))

    if not isinstance(arg, ir.Expr):
        # coerce python literal to ibis literal
        arg = ir.literal(arg)

    if not isinstance(arg, ir.ValueExpr):
        raise com.IbisTypeError('Given argument with type {} is not a value '
                                'expression'.format(type(arg)))

    if dt.issubtype(arg.type(), dtype):  # TODO: remove this, should not be required
        return arg  # subtype of expected
    if dt.castable(arg.type(), dtype):
        return arg  # implicitly castable
    else:
        raise com.IbisTypeError('Given argument with datatype {} is not '
                                'subtype of {} nor implicitly castable to '
                                'it'.format(arg.type(), dtype))


@validator
def scalar(inner, arg):
    return instanceof(ir.ScalarExpr, inner(arg))


@validator
def column(inner, arg):
    return instanceof(ir.ColumnExpr, inner(arg))


# TODO: change it to raise instead to locate all temporary noop validator
any = value(dt.any)
#null = instanceof(dt.Null)#value(dt.null)
double = value(dt.double)
string = value(dt.string)
boolean = value(dt.boolean)
integer = value(dt.int64)
decimal = value(dt.decimal)
floating = value(dt.float64)
date = value(dt.date)
time = value(dt.time)
timestamp = value(dt.timestamp)
category = value(dt.category)
# TODO: previouse number rules allowed booleans by default
temporal = oneof([timestamp, date, time])
numeric = oneof([integer, floating, decimal, boolean])
strict_numeric = oneof([integer, floating, decimal])  # without boolean



@validator
def interval(arg, units=None):
    arg = value(dt.interval, arg)
    unit = arg.type().unit
    if units is not None and unit not in units:
        msg = 'Interval unit `{}` is not among the allowed ones {}'
        raise com.IbisTypeError(msg.format(unit, units))
    return arg


table = instanceof(ir.TableExpr)
schema = instanceof(sch.Schema)


@validator
def szuper(klass, arg):
    # TODO
    return instanceof(klass, arg)


def shapeof(arg, dtype=None):
    if isinstance(arg, str):
        return lambda self: shapeof(getattr(self, arg), dtype=dtype)

    if isinstance(arg, (tuple, list)):
        # FIXME
        # datatype = highest_precedence_type({a.type() for a in arg})
        columnar = util.any_of(arg, ir.AnyColumn)
    else:
        datatype = dtype or arg.type()
        columnar = isinstance(arg, ir.AnyColumn)

    dtype = dt.dtype(dtype or datatype)
    if columnar:
        return dtype.array_type()
    else:
        return dtype.scalar_type()


def scalarof(name):
    def output_type(self):
        arg = getattr(self, name)
        output_dtype = arg.type()
        return output_dtype.scalar_type()
    return output_type


def arrayof(name):
    def output_type(self):
        arg = getattr(self, name)
        output_dtype = arg.type()
        return output_dtype.array_type()
    return output_type


def typeof(name):
    def output_type(self):
        return getattr(self, name)._factory
    return output_type


def highest_precedence_type(exprs):
    # Return the highest precedence type from the passed expressions. Also
    # verifies that there are valid implicit casts between any of the types and
    # the selected highest precedence type
    if not exprs:
        raise ValueError('Must pass at least one expression')

    expr_dtypes = {expr.type() for expr in exprs}
    return dt.highest_precedence(expr_dtypes)


def comparable(left, right):
    return ir.castable(left, right) or ir.castable(right, left)
