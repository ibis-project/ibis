import enum
import functools
import operator
from itertools import product, starmap

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.common.validators import (  # noqa: F401
    immutable_property,
    instance_of,
    isin,
    list_of,
    map_to,
    one_of,
    optional,
    tuple_of,
    validator,
)


class Shape(enum.IntEnum):
    SCALAR = 0
    COLUMNAR = 1
    # TABULAR = 2


def highest_precedence_dtype(exprs):
    """Return the highest precedence type from the passed expressions

    Also verifies that there are valid implicit casts between any of the types
    and the selected highest precedence type.
    This is a thin wrapper around datatypes highest precedence check.

    Parameters
    ----------
    exprs : Iterable[ir.Value]
      A sequence of Expressions

    Returns
    -------
    dtype: DataType
      The highest precedence datatype
    """
    if not exprs:
        return dt.null

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


# ---------------------------------------------------------------------
# Input type validators / coercion functions


# TODO(kszucs): deprecate then remove
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
def value_list_of(inner, arg, **kwargs):
    # TODO(kszucs): would be nice to remove ops.ValueList
    # the main blocker is that some of the backends execution
    # model depends on the wrapper operation, for example
    # the dispatcher in pandas requires operation objects
    import ibis.expr.operations as ops

    values = tuple_of(inner, arg, **kwargs)
    return ops.ValueList(values).to_expr()


@validator
def sort_key(key, *, from_=None, this):
    import ibis.expr.operations as ops

    table = this[from_] if from_ is not None else None
    return ops.sortkeys._to_sort_key(key, table=table)


@validator
def datatype(arg, **kwargs):
    return dt.dtype(arg)


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
    arg : Value
      An ibis value expression with the specified datatype
    """
    if not isinstance(arg, ir.Expr):
        # coerce python literal to ibis literal
        arg = ir.literal(arg)

    if not isinstance(arg, ir.Value):
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
    return instance_of(ir.Scalar, inner(arg, **kwargs))


@validator
def column(inner, arg, **kwargs):
    return instance_of(ir.Column, inner(arg, **kwargs))


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
# Ouput type functions


@util.deprecated(version="4.0", instead="")
def promoter(fn):  # pragma: no cover
    @functools.wraps(fn)
    def wrapper(name_or_value, *args, **kwargs):
        if isinstance(name_or_value, str):
            return lambda self: fn(
                getattr(self, name_or_value), *args, **kwargs
            )
        else:
            return fn(name_or_value, *args, **kwargs)

    return wrapper


def dtype_like(name):
    @immutable_property
    def output_dtype(self):
        arg = getattr(self, name)
        if util.is_iterable(arg):
            return highest_precedence_dtype(arg)
        else:
            return arg.type()

    return output_dtype


def shape_like(name):
    @immutable_property
    def output_shape(self):
        arg = getattr(self, name)
        if util.is_iterable(arg):
            for expr in arg:
                try:
                    op = expr.op()
                except AttributeError:
                    continue
                if op.output_shape is Shape.COLUMNAR:
                    return Shape.COLUMNAR
            return Shape.SCALAR
        else:
            return arg.op().output_shape

    return output_shape


# TODO(kszucs): might just use bounds instead of actual literal values
# that could simplify interval binop output_type methods
# TODO(kszucs): pre-generate mapping?
def _promote_integral_binop(exprs, op):
    dtypes = []
    bounds = []
    for expr in exprs:
        try:
            bounds.append([expr.op().value])
        except AttributeError:
            dtypes.append(expr.type())
            bounds.append(expr.type().bounds)

    all_unsigned = dtypes and util.all_of(dtypes, dt.UnsignedInteger)
    # In some cases, the bounding type might be int8, even though neither
    # of the types are that small. We want to ensure the containing type is
    # _at least_ as large as the smallest type in the expression.
    values = list(starmap(op, product(*bounds)))
    dtypes.extend(dt.infer(v, prefer_unsigned=all_unsigned) for v in values)
    return dt.highest_precedence(dtypes)


def _promote_decimal_dtype(args, op):

    if len(args) != 2:
        return highest_precedence_dtype(args)

    # TODO: Add support for setting the maximum precision and maximum scale
    lhs_prec = args[0].type().precision
    lhs_scale = args[0].type().scale
    rhs_prec = args[1].type().precision
    rhs_scale = args[1].type().scale
    max_prec = 31 if lhs_prec <= 31 and rhs_prec <= 31 else 63
    max_scale = 31

    if op is operator.mul:
        return dt.Decimal(
            min(max_prec, lhs_prec + rhs_prec),
            min(max_scale, lhs_scale + rhs_scale),
        )
    if op is operator.add or op is operator.sub:
        return dt.Decimal(
            min(
                max_prec,
                max(
                    lhs_prec - lhs_scale,
                    rhs_prec - rhs_scale,
                )
                + max(lhs_scale, rhs_scale)
                + 1,
            ),
            max(lhs_scale, rhs_scale),
        )
    return highest_precedence_dtype(args)


def numeric_like(name, op):
    @immutable_property
    def output_dtype(self):
        args = getattr(self, name)
        if util.all_of(args, ir.IntegerValue):
            result = _promote_integral_binop(args, op)
        elif util.all_of(args, ir.DecimalValue):
            result = _promote_decimal_dtype(args, op)
        else:
            result = highest_precedence_dtype(args)

        return result

    return output_dtype


@validator
def table(arg, *, schema=None, **kwargs):
    """A table argument.

    Parameters
    ----------
    schema
        A validator for the table's columns. Only column subset validators are
        currently supported. Accepts any arguments that `sch.schema` accepts.
        See the example for usage.
    arg
        An argument

    The following op will accept an argument named `'table'`. Note that the
    `schema` argument specifies rules for columns that are required to be in
    the table: `time`, `group` and `value1`. These must match the types
    specified in the column rules. Column `value2` is optional, but if present
    it must be of the specified type. The table may have extra columns not
    specified in the schema.
    """
    import ibis

    if not isinstance(arg, ir.Table):
        try:
            return ibis.table(data=arg, schema=schema)
        except Exception as e:
            raise com.IbisTypeError(
                f'Argument is not a table; got type {type(arg).__name__}'
            ) from e

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
    if name not in this:
        raise com.IbisTypeError(f"Could not get table {name} from {this}")
    table = this[name]

    if isinstance(column, (str, int)):
        return table[column]
    elif isinstance(column, ir.Column):
        if not column.has_name():
            raise com.IbisTypeError(f"Passed column {column} has no name")

        maybe_column = column.get_name()
        try:
            if column.equals(table[maybe_column]):
                return column
            else:
                raise com.IbisTypeError(
                    f"Passed column is not a column in {type(table)}"
                )
        except com.IbisError:
            raise com.IbisTypeError(
                f"Cannot get column {maybe_column} from {type(table)}"
            )

    raise com.IbisTypeError(
        "value must be an int or str or Column, got "
        f"{type(column).__name__}"
    )


@validator
def base_table_of(name, *, this):
    from ibis.expr.analysis import find_first_base_table

    arg = this[name]
    base = find_first_base_table(arg)
    if base is None:
        raise com.IbisTypeError(f"`{arg}` doesn't have a base table")
    else:
        return base


@validator
def function_of(
    arg,
    fn,
    *,
    output_rule=any,
    this,
):
    if not util.is_function(fn):
        raise com.IbisTypeError(
            'argument `fn` must be a function, lambda or deferred operation'
        )

    if isinstance(arg, str):
        arg = this[arg]
    elif callable(arg):
        arg = arg(this=this)

    return output_rule(fn(arg), this=this)


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
@util.deprecated(version="4.0", instead="")
def is_computable_input(value, **kwargs):  # pragma: no cover
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
@util.deprecated(version="4.0.0", instead="")
def named_literal(value, **kwargs):
    import ibis.expr.operations as ops

    if not isinstance(value, ir.Scalar):
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
    from ibis.expr.analysis import find_first_base_table
    from ibis.expr.window import Window

    if not isinstance(win, Window):
        raise com.IbisTypeError(
            "`win` argument should be of type `ibis.expr.window.Window`; "
            f"got type {type(win).__name__}"
        )

    table = find_first_base_table(this[from_base_table_of])
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
