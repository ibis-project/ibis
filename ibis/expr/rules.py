import enum
import functools
from contextlib import suppress
from itertools import product, starmap

from toolz import compose, curry, identity

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.signature as sig
import ibis.expr.types as ir
import ibis.util as util

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

    raise com.IbisTypeError(
        "argument passes none of the following rules: "
        f"{', '.join(map(repr, inners))}"
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
def map_to(mapping, variant, **kwargs):
    return mapping[variant]


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
def container_of(inner, arg, *, type, min_length=0, flatten=False, **kwargs):
    if not util.is_iterable(arg):
        raise com.IbisTypeError('Argument must be a sequence')

    if len(arg) < min_length:
        raise com.IbisTypeError(
            f'Arg must have at least {min_length} number of elements'
        )

    if flatten:
        arg = util.flatten_iterable(arg)

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
def sort_key(key, *, from_=None, this):
    import ibis.expr.operations as ops

    table = getattr(this, from_) if from_ is not None else None
    return ops.sortkeys._to_sort_key(key, table=table)


@validator
def datatype(arg, **kwargs):
    return dt.dtype(arg)


@validator
def instance_of(klasses, arg, **kwargs):
    """Require that a value has a particular Python type."""
    if not isinstance(arg, klasses):
        raise com.IbisTypeError(
            f'Given argument with type {type(arg)} '
            f'is not an instance of {klasses}'
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
            f'Given argument with type {type(arg)} is not a value '
            'expression'
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
            f'Given argument with datatype {arg.type()} is not '
            f'subtype of {dtype} nor implicitly castable to it'
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
            'Argument must be an array, '
            f'got expression which is of type {val.type()}'
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
def table(arg, *, schema=None, **kwargs):
    """A table argument.

    Parameters
    ----------
    schema : Union[sch.Schema, List[Tuple[str, dt.DataType], None]
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
    if not isinstance(arg, ir.TableExpr):
        raise com.IbisTypeError(
            f'Argument is not a table; got type {type(arg).__name__}'
        )

    if schema is not None:
        if arg.schema() >= sch.schema(schema):
            return arg

        raise com.IbisTypeError(
            f'Argument is not a table with column subset of {schema}'
        )
    return arg


@validator
def column_from(name, column, *, this):
    """A column from a named table.

    This validator accepts columns passed as string, integer, or column
    expression. In the case of a column expression, this validator
    checks if the column in the table is equal to the column being
    passed.
    """
    if not hasattr(this, name):
        raise com.IbisTypeError(f"Could not get table {name} from {this}")
    table = getattr(this, name)

    if isinstance(column, (str, int)):
        return table[column]
    elif isinstance(column, ir.AnyColumn):
        if not column.has_name():
            raise com.IbisTypeError(f"Passed column {column} has no name")

        maybe_column = column.get_name()
        try:
            if column.equals(table[maybe_column]):
                return column
            else:
                raise com.IbisTypeError(
                    f"Passed column is not a column in {table}"
                )
        except com.IbisError:
            raise com.IbisTypeError(
                f"Cannot get column {maybe_column} from {table}"
            )

    raise com.IbisTypeError(
        "value must be an int or str or AnyColumn, got "
        f"{type(column).__name__}"
    )


@validator
def function_of(
    argument,
    fn,
    *,
    output_rule=any,
    preprocess=identity,
    this,
):
    if not util.is_function(fn):
        raise com.IbisTypeError('argument `fn` must be a function or lambda')

    return output_rule(fn(preprocess(getattr(this, argument))), this=this)


@validator
def reduction(argument, **kwargs):
    from ibis.expr.analysis import is_reduction

    if not is_reduction(argument):
        raise com.IbisTypeError("`argument` must be a reduction")

    return argument


@validator
def non_negative_integer(arg, **kwargs):
    if not isinstance(arg, int):
        raise com.IbisTypeError(
            f"positive integer must be int type, got {type(arg).__name__}"
        )
    if arg < 0:
        raise ValueError("got negative value for non-negative integer rule")
    return arg


@validator
def python_literal(value, arg, **kwargs):
    if (
        not isinstance(arg, type(value))
        or not isinstance(value, type(arg))
        or arg != value
    ):
        raise ValueError(
            "arg must be a literal exactly equal in type and value to value "
            f"{value} with type {type(value)}, got `arg` with type {type(arg)}"
        )
    return arg


@validator
def is_computable_input(value, **kwargs):
    from ibis.backends.pandas.core import (
        is_computable_input as _is_computable_input,
    )

    if not _is_computable_input(value):
        raise com.IbisTypeError(
            f"object {value} is not a computable input; "
            "did you register the type with "
            "ibis.backends.pandas.core.is_computable_input?"
        )
    return value


@validator
def named_literal(value, **kwargs):
    import ibis.expr.operations as ops

    if not isinstance(value, ir.ScalarExpr):
        raise com.IbisTypeError(
            "`value` must be a scalar expression; "
            f"got value of type {type(value).__name__}"
        )

    if not isinstance(value.op(), ops.Literal):
        raise com.IbisTypeError(
            "`value` must map to an ibis literal; "
            f"got expr with op {type(value.op()).__name__}"
        )

    # check that the literal has a name
    if not value.has_name():
        raise com.IbisTypeError("`value` literal is not named")

    return value


@validator
def pair(inner_left, inner_right, a, b, **kwargs):
    return inner_left(a, **kwargs), inner_right(b, **kwargs)


@validator
def analytic(arg, **kwargs):
    from ibis.expr.analysis import is_analytic

    if not is_analytic(arg):
        raise com.IbisInputError(
            'Expression does not contain a valid window operation'
        )
    return arg


@validator
def window(win, *, from_base_table_of, this):
    from ibis.expr.window import Window

    if not isinstance(win, Window):
        raise com.IbisTypeError(
            "`win` argument should be of type `ibis.expr.window.Window`; "
            f"got type {type(win).__name__}"
        )
    table = ir.find_base_table(getattr(this, from_base_table_of))
    if table is not None:
        win = win.bind(table)

    if win.max_lookback is not None:
        error_msg = (
            "'max lookback' windows must be ordered " "by a timestamp column"
        )
        if len(win._order_by) != 1:
            raise com.IbisInputError(error_msg)
        order_var = win._order_by[0].op().args[0]
        if not isinstance(order_var.type(), dt.Timestamp):
            raise com.IbisInputError(error_msg)
    return win


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
