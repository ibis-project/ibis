import enum
import functools
from contextlib import suppress
from itertools import product, starmap

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.signature as sig
import ibis.expr.types as ir
import ibis.util as util

try:
    from cytoolz import compose, curry
except ImportError:
    from toolz import compose, curry


optional = sig.Optional


class validator(curry, sig.Validator):
    def __repr__(self):
        return '{}({}{})'.format(
            self.func.__name__,
            repr(self.args)[1:-1],
            ', '.join(f'{k}={v!r}' for k, v in self.keywords.items()),
        )


def highest_precedence_dtype(exprs):
    """Return the highest precedence type from the passed expressions

    Also verifies that there are valid implicit casts between any of the types
    and the selected highest precedence type.
    This is a thin wrapper around datatypes highest precedence check.

    Parameters
    ----------
    exprs : Iterable[ir.ValueExpr]
      A sequence of Expressions

    Returns
    -------
    dtype: DataType
      The highest precedence datatype
    """
    if not exprs:
        raise ValueError('Must pass at least one expression')

    return dt.highest_precedence(expr.type() for expr in exprs)


def castable(source, target):
    """Return whether source ir type is implicitly castable to target

    Based on the underlying datatypes and the value in case of Literals
    """
    op = source.op()
    value = getattr(op, 'value', None)
    return dt.castable(source.type(), target.type(), value=value)


def comparable(left, right):
    return castable(left, right) or castable(right, left)


def cast(source, target):
    """Currently Literal to *Scalar implicit casts are allowed"""
    import ibis.expr.operations as ops  # TODO: don't use ops here

    if not castable(source, target):
        raise com.IbisTypeError('Source is not castable to target type!')

    # currently it prevents column -> scalar implicit castings
    # however the datatypes are matching
    op = source.op()
    if not isinstance(op, ops.Literal):
        raise com.IbisTypeError('Only able to implicitly cast literals!')

    out_type = target.type().scalar_type()
    return out_type(op)


# ---------------------------------------------------------------------
# Input type validators / coercion functions


@validator
def noop(arg, **kwargs):
    return arg


@validator
def one_of(inners, arg, **kwargs):
    """At least one of the inner validators must pass"""
    for inner in inners:
        with suppress(com.IbisTypeError, ValueError):
            return inner(arg, **kwargs)

    rules_formatted = ', '.join(map(repr, inners))
    raise com.IbisTypeError(
        f'Arg passes neither of the following rules: {rules_formatted}'
    )


@validator
def all_of(inners, arg, *, this):
    """All of the inner validators must pass.

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
      Value maybe coerced by inner validators to the appropiate types
    """
    return compose(*inners)(arg, this=this)


@validator
def isin(values, arg, **kwargs):
    if arg not in values:
        raise ValueError(f'Value with type {type(arg)} is not in {values!r}')
    if isinstance(values, dict):  # TODO check for mapping instead
        return values[arg]
    else:
        return arg


@validator
def member_of(obj, arg, **kwargs):
    if isinstance(arg, ir.EnumValue):
        arg = arg.op().value
    if isinstance(arg, enum.Enum):
        enum.unique(obj)  # check that enum has unique values
        arg = arg.name

    if not hasattr(obj, arg):
        raise com.IbisTypeError(
            f'Value with type {type(arg)} is not a member of {obj}'
        )
    return getattr(obj, arg)


@validator
def container_of(inner, arg, *, type, min_length=0, **kwargs):
    if not util.is_iterable(arg):
        raise com.IbisTypeError('Argument must be a sequence')

    if len(arg) < min_length:
        raise com.IbisTypeError(
            f'Arg must have at least {min_length} number of elements'
        )
    return type(inner(item, **kwargs) for item in arg)


list_of = container_of(type=list)
tuple_of = container_of(type=tuple)


@validator
def value_list_of(inner, arg, **kwargs):
    # TODO(kszucs): would be nice to remove ops.ValueList
    # the main blocker is that some of the backends execution
    # model depends on the wrapper operation, for example
    # the dispatcher in pandas requires operation objects
    import ibis.expr.operations as ops

    values = list_of(inner, arg, **kwargs)
    return ops.ValueList(values).to_expr()


@validator
def datatype(arg, **kwargs):
    return dt.dtype(arg)


@validator
def instance_of(klass, arg, **kwargs):
    """Require that a value has a particular Python type."""
    if not isinstance(arg, klass):
        raise com.IbisTypeError(
            'Given argument with type {} is not an instance of {}'.format(
                type(arg), klass
            )
        )
    return arg


@validator
def coerce_to(klass, arg, **kwargs):
    return klass(arg)


@validator
def value(dtype, arg, **kwargs):
    """Validates that the given argument is a Value with a particular datatype

    Parameters
    ----------
    dtype : DataType subclass or DataType instance
    arg : python literal or an ibis expression
      If a python literal is given the validator tries to coerce it to an ibis
      literal.

    Returns
    -------
    arg : AnyValue
      An ibis value expression with the specified datatype
    """
    if not isinstance(arg, ir.Expr):
        # coerce python literal to ibis literal
        arg = ir.literal(arg)

    if not isinstance(arg, ir.AnyValue):
        raise com.IbisTypeError(
            'Given argument with type {} is not a value '
            'expression'.format(type(arg))
        )

    # retrieve literal values for implicit cast check
    value = getattr(arg.op(), 'value', None)

    if isinstance(dtype, type) and isinstance(arg.type(), dtype):
        # dtype class has been specified like dt.Interval or dt.Decimal
        return arg
    elif dt.castable(arg.type(), dt.dtype(dtype), value=value):
        # dtype instance or string has been specified and arg's dtype is
        # implicitly castable to it, like dt.int8 is castable to dt.int64
        return arg
    else:
        raise com.IbisTypeError(
            'Given argument with datatype {} is not '
            'subtype of {} nor implicitly castable to '
            'it'.format(arg.type(), dtype)
        )


@validator
def scalar(inner, arg, **kwargs):
    return instance_of(ir.ScalarExpr, inner(arg, **kwargs))


@validator
def column(inner, arg, **kwargs):
    return instance_of(ir.ColumnExpr, inner(arg, **kwargs))


@validator
def array_of(inner, arg, **kwargs):
    val = arg if isinstance(arg, ir.Expr) else ir.literal(arg)
    argtype = val.type()
    if not isinstance(argtype, dt.Array):
        raise com.IbisTypeError(
            'Argument must be an array, got expression {} which is of type '
            '{}'.format(val, val.type())
        )
    value_dtype = inner(val[0], **kwargs).type()
    array_dtype = dt.Array(value_dtype)
    return value(array_dtype, val, **kwargs)


any = value(dt.any)
double = value(dt.double)
string = value(dt.string)
boolean = value(dt.boolean)
integer = value(dt.int64)
decimal = value(dt.Decimal)
floating = value(dt.float64)
date = value(dt.date)
time = value(dt.time)
timestamp = value(dt.Timestamp)
category = value(dt.category)
temporal = one_of([timestamp, date, time])

strict_numeric = one_of([integer, floating, decimal])
soft_numeric = one_of([integer, floating, decimal, boolean])
numeric = soft_numeric

set_ = value(dt.Set)
array = value(dt.Array)
struct = value(dt.Struct)
mapping = value(dt.Map(dt.any, dt.any))

geospatial = value(dt.GeoSpatial)
point = value(dt.Point)
linestring = value(dt.LineString)
polygon = value(dt.Polygon)
multilinestring = value(dt.MultiLineString)
multipoint = value(dt.MultiPoint)
multipolygon = value(dt.MultiPolygon)


@validator
def interval(arg, units=None, **kwargs):
    arg = value(dt.Interval, arg)
    unit = arg.type().unit
    if units is not None and unit not in units:
        msg = 'Interval unit `{}` is not among the allowed ones {}'
        raise com.IbisTypeError(msg.format(unit, units))
    return arg


@validator
def client(arg, **kwargs):
    from ibis.backends.base import BaseBackend

    return instance_of(BaseBackend, arg)


# ---------------------------------------------------------------------
# Ouput type promoter functions


def promoter(fn):
    @functools.wraps(fn)
    def wrapper(name_or_value, *args, **kwargs):
        if isinstance(name_or_value, str):
            return lambda self: fn(
                getattr(self, name_or_value), *args, **kwargs
            )
        else:
            return fn(name_or_value, *args, **kwargs)

    return wrapper


@promoter
def shape_like(arg, dtype=None):
    if isinstance(arg, (tuple, list, ir.ListExpr)):
        datatype = dtype or highest_precedence_dtype(arg)
        columnar = util.any_of(arg, ir.AnyColumn)
    else:
        datatype = dtype or arg.type()
        columnar = isinstance(arg, ir.AnyColumn)

    dtype = dt.dtype(datatype)

    if columnar:
        return dtype.column_type()
    else:
        return dtype.scalar_type()


@promoter
def scalar_like(arg):
    output_dtype = arg.type()
    return output_dtype.scalar_type()


@promoter
def array_like(arg):
    output_dtype = arg.type()
    return output_dtype.column_type()


column_like = array_like


@promoter
def typeof(arg):
    return arg._factory


@validator
def table(schema, arg, **kwargs):
    """A table argument.

    Parameters
    ----------
    schema : Union[sch.Schema, List[Tuple[str, dt.DataType]]
        A validator for the table's columns. Only column subset validators are
        currently supported. Accepts any arguments that `sch.schema` accepts.
        See the example for usage.
    arg : The validatable argument.

    Examples
    --------
    The following op will accept an argument named ``'table'``. Note that the
    ``schema`` argument specifies rules for columns that are required to be in
    the table: ``time``, ``group`` and ``value1``. These must match the types
    specified in the column rules. Column ``value2`` is optional, but if
    present it must be of the specified type. The table may have extra columns
    not specified in the schema.
    """
    assert isinstance(arg, ir.TableExpr)

    if arg.schema() >= sch.schema(schema):
        return arg

    raise com.IbisTypeError(
        f'Argument is not a table with column subset of {schema}'
    )


# TODO: might just use bounds instead of actual literal values
# that could simplify interval binop output_type methods
def _promote_numeric_binop(exprs, op):
    bounds, dtypes = [], []
    for arg in exprs:
        dtypes.append(arg.type())
        if hasattr(arg.op(), 'value'):
            # arg.op() is a literal
            bounds.append([arg.op().value])
        else:
            bounds.append(arg.type().bounds)

    # In some cases, the bounding type might be int8, even though neither
    # of the types are that small. We want to ensure the containing type is
    # _at least_ as large as the smallest type in the expression.
    values = starmap(op, product(*bounds))
    dtypes += [dt.infer(value, allow_overflow=True) for value in values]

    return dt.highest_precedence(dtypes)


@promoter
def numeric_like(args, op):
    if util.all_of(args, ir.IntegerValue):
        dtype = _promote_numeric_binop(args, op)
        return shape_like(args, dtype=dtype)
    else:
        return shape_like(args)


# TODO: create varargs marker for impala udfs
