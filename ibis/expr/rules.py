from __future__ import annotations

import enum
import operator
from itertools import product, starmap

from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.common.annotations import attribute, optional
from ibis.common.validators import (
    bool_,
    callable_with,  # noqa: F401
    coerced_to,  # noqa: F401
    equal_to,  # noqa: F401
    instance_of,
    isin,
    lazy_instance_of,
    map_to,
    one_of,
    option,  # noqa: F401
    pair_of,  # noqa: F401
    ref,
    str_,
    tuple_of,
    validator,
)
from ibis.expr.deferred import Deferred


# TODO(kszucs): consider to rename to datashape
@public
class Shape(enum.IntEnum):
    SCALAR = 0
    COLUMNAR = 1
    # TABULAR = 2

    def is_scalar(self):
        return self is Shape.SCALAR

    def is_columnar(self):
        return self is Shape.COLUMNAR


@public
def highest_precedence_shape(nodes):
    return max(node.output_shape for node in nodes)


@public
def highest_precedence_dtype(nodes):
    """Return the highest precedence type from the passed expressions.

    Also verifies that there are valid implicit casts between any of the types
    and the selected highest precedence type.
    This is a thin wrapper around datatypes highest precedence check.

    Parameters
    ----------
    nodes : Iterable[ops.Value]
      A sequence of Expressions

    Returns
    -------
    dtype: DataType
      The highest precedence datatype
    """
    return dt.highest_precedence(node.output_dtype for node in nodes)


@public
def castable(source, target):
    """Return whether source ir type is implicitly castable to target.

    Based on the underlying datatypes and the value in case of Literals
    """
    value = getattr(source, 'value', None)
    return dt.castable(source.output_dtype, target.output_dtype, value=value)


@public
def comparable(left, right):
    return castable(left, right) or castable(right, left)


class rule(validator):
    def _erase_expr(self, value):
        return value.op() if isinstance(value, ir.Expr) else value

    def __call__(self, *args, **kwargs):
        args = map(self._erase_expr, args)
        kwargs = {k: self._erase_expr(v) for k, v in kwargs.items()}
        result = super().__call__(*args, **kwargs)
        assert not isinstance(result, ir.Expr)
        return result


# ---------------------------------------------------------------------
# Input type validators / coercion functions


@validator
def expr_of(inner, value, **kwargs):
    value = inner(value, **kwargs)
    return value if isinstance(value, ir.Expr) else value.to_expr()


@rule
def just(arg):
    return lambda **_: arg


@rule
def sort_key_from(table_ref, key, **kwargs):
    import ibis.expr.operations as ops

    is_ascending = {
        "asc": True,
        "ascending": True,
        "desc": False,
        "descending": False,
        0: False,
        1: True,
        False: False,
        True: True,
    }

    if callable(key):
        key = function_of(table_ref, key, **kwargs)

    if isinstance(key, ops.SortKey):
        return key
    elif isinstance(key, tuple):
        key, order = key
    else:
        key, order = key, True

    if isinstance(key, (str, int)):
        # Actual str/int keys must refer to columns in the table, we don't want
        # to fallback to converting them to expressions with ibis.literal
        key = column_from(table_ref, key, **kwargs)
    else:
        key = one_of(
            (function_of(table_ref), column_from(table_ref), any),
            key,
            **kwargs,
        )

    if isinstance(order, str):
        order = order.lower()
    order = map_to(is_ascending, order)

    return ops.SortKey(key, ascending=order)


@rule
def datatype(arg, **kwargs):
    return dt.dtype(arg)


# TODO(kszucs): make type argument the first and mandatory, similarly to the
# value rule, move out the type inference to `ir.literal()` method
# TODO(kszucs): may not make sense to support an explicit datatype here, we
# could do the coercion in the API function ibis.literal()
@rule
def literal(dtype, value, **kwargs):
    import ibis.expr.operations as ops

    if isinstance(value, ops.Literal):
        return value

    try:
        inferred_dtype = dt.infer(value)
    except com.InputTypeError:
        has_inferred = False
    else:
        has_inferred = True

    if dtype is None:
        has_explicit = False
    else:
        has_explicit = True
        # TODO(kszucs): handle class-like dtype definitions here explicitly
        explicit_dtype = dt.dtype(dtype)

    if has_explicit and has_inferred:
        try:
            # ensure type correctness: check that the inferred dtype is
            # implicitly castable to the explicitly given dtype and value
            dtype = dt.cast(inferred_dtype, target=explicit_dtype, value=value)
        except com.IbisTypeError:
            raise TypeError(f'Value {value!r} cannot be safely coerced to `{dtype}`')
    elif has_explicit:
        dtype = explicit_dtype
    elif has_inferred:
        dtype = inferred_dtype
    else:
        raise com.IbisTypeError(
            'The datatype of value {!r} cannot be inferred, try '
            'passing it explicitly with the `type` keyword.'.format(value)
        )

    if dtype.is_null():
        return ops.NullLiteral()

    value = dt.normalize(dtype, value)
    return ops.Literal(value, dtype=dtype)


@rule
def value(dtype, arg, **kwargs):
    """Validates that the given argument is a Value with a particular datatype.

    Parameters
    ----------
    dtype
        DataType subclass or DataType instance
    arg
        If a Python literal is given the validator tries to coerce it to an ibis
        literal.
    kwargs
        Keyword arguments

    Returns
    -------
    ir.Value
        An ibis value expression with the specified datatype
    """
    import ibis.expr.operations as ops

    if isinstance(arg, Deferred):
        raise com.IbisTypeError(
            "Deferred input is not allowed, try passing a lambda function instead. "
            "For example, instead of writing `f(_.a)` write `lambda t: f(t.a)`"
        )

    if not isinstance(arg, ops.Value):
        # coerce python literal to ibis literal
        arg = literal(None, arg)

    if dtype is None:
        # no datatype restriction
        return arg
    elif isinstance(dtype, type):
        # dtype class has been specified like dt.Interval or dt.Decimal
        if not issubclass(dtype, dt.DataType):
            raise com.IbisTypeError(
                f"Datatype specification {dtype} is not a subclass dt.DataType"
            )
        elif isinstance(arg.output_dtype, dtype):
            return arg
        else:
            raise com.IbisTypeError(
                f'Given argument with datatype {arg.output_dtype} is not '
                f'subtype of {dtype}'
            )
    elif isinstance(dtype, (dt.DataType, str)):
        # dtype instance or string has been specified and arg's dtype is
        # implicitly castable to it, like dt.int8 is castable to dt.int64
        dtype = dt.dtype(dtype)
        # retrieve literal values for implicit cast check
        value = getattr(arg, 'value', None)
        if dt.castable(arg.output_dtype, dtype, value=value):
            return arg
        else:
            raise com.IbisTypeError(
                f'Given argument with datatype {arg.output_dtype} is not '
                f'implicitly castable to {dtype}'
            )
    else:
        raise com.IbisTypeError(f'Invalid datatype specification {dtype}')


@rule
def scalar(inner, arg, **kwargs):
    arg = inner(arg, **kwargs)
    if arg.output_shape.is_scalar():
        return arg
    else:
        raise com.IbisTypeError(f"{arg} it not a scalar")


@rule
def column(inner, arg, **kwargs):
    arg = inner(arg, **kwargs)
    if arg.output_shape.is_columnar():
        return arg
    else:
        raise com.IbisTypeError(f"{arg} it not a column")


any = value(None)
double = value(dt.double)
string = value(dt.string)
boolean = value(dt.boolean)
integer = value(dt.int64)
decimal = value(dt.Decimal)
floating = value(dt.float64)
date = value(dt.date)
time = value(dt.time)
timestamp = value(dt.Timestamp)
temporal = one_of([timestamp, date, time])
json = value(dt.json)

strict_numeric = one_of([integer, floating, decimal])
soft_numeric = one_of([integer, floating, decimal, boolean])
numeric = soft_numeric

set_ = value(dt.Set)
array = value(dt.Array)
struct = value(dt.Struct)
mapping = value(dt.Map)

geospatial = value(dt.GeoSpatial)
point = value(dt.Point)
linestring = value(dt.LineString)
polygon = value(dt.Polygon)
multilinestring = value(dt.MultiLineString)
multipoint = value(dt.MultiPoint)
multipolygon = value(dt.MultiPolygon)


@public
@rule
def interval(arg, units=None, **kwargs):
    arg = value(dt.Interval, arg)
    unit = arg.output_dtype.unit
    if units is not None and unit not in units:
        msg = 'Interval unit `{}` is not among the allowed ones {}'
        raise com.IbisTypeError(msg.format(unit, units))
    return arg


@public
@rule
def client(arg, **kwargs):
    from ibis.backends.base import BaseBackend

    return instance_of(BaseBackend, arg)


# ---------------------------------------------------------------------
# Ouput type functions


@public
def dtype_like(name):
    @attribute.default
    def output_dtype(self):
        args = getattr(self, name)
        args = args if util.is_iterable(args) else [args]
        return highest_precedence_dtype(args)

    return output_dtype


@public
def shape_like(name):
    @attribute.default
    def output_shape(self):
        args = getattr(self, name)
        args = args if util.is_iterable(args) else [args]
        return highest_precedence_shape(args)

    return output_shape


# TODO(kszucs): might just use bounds instead of actual literal values
# that could simplify interval binop output_type methods
# TODO(kszucs): pre-generate mapping?


def _promote_integral_binop(exprs, op):
    import ibis.expr.operations as ops

    bounds, dtypes = [], []
    for arg in exprs:
        dtypes.append(arg.output_dtype)
        if isinstance(arg, ops.Literal):
            bounds.append([arg.value])
        else:
            bounds.append(arg.output_dtype.bounds)

    all_unsigned = dtypes and util.all_of(dtypes, dt.UnsignedInteger)
    # In some cases, the bounding type might be int8, even though neither
    # of the types are that small. We want to ensure the containing type is
    # _at least_ as large as the smallest type in the expression.
    values = starmap(op, product(*bounds))
    dtypes += [dt.infer(v, prefer_unsigned=all_unsigned) for v in values]

    return dt.highest_precedence(dtypes)


def _promote_decimal_binop(args, op):
    if len(args) != 2:
        return highest_precedence_dtype(args)

    # TODO: Add support for setting the maximum precision and maximum scale
    left = args[0].output_dtype
    right = args[1].output_dtype

    max_prec = 31 if left.precision <= 31 and right.precision <= 31 else 63
    max_scale = 31

    if op is operator.mul:
        return dt.Decimal(
            min(max_prec, left.precision + right.precision),
            min(max_scale, left.scale + right.scale),
        )
    elif op is operator.add or op is operator.sub:
        return dt.Decimal(
            min(
                max_prec,
                max(
                    left.precision - left.scale,
                    right.precision - right.scale,
                )
                + max(left.scale, right.scale)
                + 1,
            ),
            max(left.scale, right.scale),
        )
    else:
        return highest_precedence_dtype(args)


@public
def numeric_like(name, op):
    @attribute.default
    def output_dtype(self):
        args = getattr(self, name)
        dtypes = [arg.output_dtype for arg in args]
        if util.all_of(dtypes, dt.Integer):
            result = _promote_integral_binop(args, op)
        elif util.all_of(dtypes, dt.Decimal):
            result = _promote_decimal_binop(args, op)
        else:
            result = highest_precedence_dtype(args)

        return result

    return output_dtype


# TODO(kszucs): it could be as simple as rlz.instance_of(ops.TableNode)
# we have a single test case testing the schema superset condition, not
# used anywhere else
@public
@rule
def table(arg, schema=None, **kwargs):
    """A table argument.

    Parameters
    ----------
    arg
        A table node
    schema
        A validator for the table's columns. Only column subset validators are
        currently supported. Accepts any arguments that `sch.schema` accepts.
        See the example for usage.
    kwargs
        Keyword arguments

    The following op will accept an argument named `'table'`. Note that the
    `schema` argument specifies rules for columns that are required to be in
    the table: `time`, `group` and `value1`. These must match the types
    specified in the column rules. Column `value2` is optional, but if present
    it must be of the specified type. The table may have extra columns not
    specified in the schema.
    """
    import pandas as pd

    import ibis
    import ibis.expr.operations as ops

    if isinstance(arg, pd.DataFrame):
        arg = ibis.memtable(arg).op()

    if not isinstance(arg, ops.TableNode):
        raise com.IbisTypeError(
            f'Argument is not a table; got type {type(arg).__name__}'
        )

    if schema is not None:
        if arg.schema >= sch.schema(schema):
            return arg

        raise com.IbisTypeError(
            f'Argument is not a table with column subset of {schema}'
        )
    return arg


@public
@rule
def column_from(table_ref, column, **kwargs):
    """A column from a named table.

    This validator accepts columns passed as string, integer, or column
    expression. In the case of a column expression, this validator
    checks if the column in the table is equal to the column being
    passed.
    """
    import ibis.expr.operations as ops

    # TODO(kszucs): should avoid converting to TableExpr
    table = table_ref(**kwargs).to_expr()

    # TODO(kszucs): should avoid converting to a ColumnExpr
    if isinstance(column, ops.Node):
        column = column.to_expr()

    column = table._ensure_expr(column)

    if not isinstance(column, ir.Column):
        raise com.IbisTypeError(
            f"value must be an int or str or Column, got {type(column).__name__}"
        )

    if not column.has_name():
        raise com.IbisTypeError(f"Passed column {column} has no name")

    maybe_column = column.get_name()
    try:
        if column.equals(table[maybe_column]):
            return column.op()
        else:
            raise com.IbisTypeError(f"Passed column is not a column in {type(table)}")
    except com.IbisError:
        raise com.IbisTypeError(f"Cannot get column {maybe_column} from {type(table)}")


@public
@rule
def base_table_of(table_ref, *, this, strict=True):
    from ibis.expr.analysis import find_first_base_table

    arg = table_ref(this=this)
    base = find_first_base_table(arg)
    if strict and base is None:
        raise com.IbisTypeError(f"`{arg}` doesn't have a base table")
    return base


@public
@rule
def function_of(table_ref, fn, *, output_rule=any, this=None):
    arg = table_ref(this=this).to_expr()

    if util.is_function(fn):
        arg = fn(arg)
    elif isinstance(fn, Deferred):
        arg = fn.resolve(arg)
    else:
        raise com.IbisTypeError(
            'argument `fn` must be a function, lambda or deferred operation'
        )

    return output_rule(arg, this=this)


@public
@rule
def reduction(arg, **kwargs):
    from ibis.expr.analysis import is_reduction

    if not is_reduction(arg):
        raise com.IbisTypeError("`argument` must be a reduction")

    return arg


@public
@rule
def analytic(arg, **kwargs):
    from ibis.expr.analysis import is_analytic

    if not is_analytic(arg):
        raise com.IbisInputError('Expression does not contain a valid window operation')

    return arg


@public
@rule
def window_boundary(inner, arg, **kwargs):
    import ibis.expr.operations as ops

    arg = inner(arg, **kwargs)

    if isinstance(arg, ops.WindowBoundary):
        return arg
    elif isinstance(arg, ops.Negate):
        return ops.WindowBoundary(arg.arg, preceding=True)
    elif isinstance(arg, ops.Literal):
        new = arg.copy(value=abs(arg.value))
        return ops.WindowBoundary(new, preceding=arg.value < 0)
    elif isinstance(arg, ops.Value):
        return ops.WindowBoundary(arg, preceding=False)
    else:
        raise TypeError(f'Invalid window boundary type: {type(arg)}')


row_window_boundary = window_boundary(integer)
range_window_boundary = window_boundary(one_of([numeric, interval]))


def _arg_type_error_format(op):
    from ibis.expr.operations.generic import Literal

    if isinstance(op, Literal):
        return f"Literal({op.value}):{op.output_dtype}"
    else:
        return f"{op.name}:{op.output_dtype}"


public(
    any=any,
    array=array,
    bool=bool_,
    boolean=boolean,
    date=date,
    decimal=decimal,
    double=double,
    floating=floating,
    geospatial=geospatial,
    integer=integer,
    isin=isin,
    json=json,
    lazy_instance_of=lazy_instance_of,
    linestring=linestring,
    mapping=mapping,
    multilinestring=multilinestring,
    multipoint=multipoint,
    numeric=numeric,
    optional=optional,
    point=point,
    polygon=polygon,
    ref=ref,
    set_=set_,
    soft_numeric=soft_numeric,
    str_=str_,
    strict_numeric=strict_numeric,
    string=string,
    struct=struct,
    temporal=temporal,
    time=time,
    timestamp=timestamp,
    tuple_of=tuple_of,
    row_window_boundary=row_window_boundary,
    range_window_boundary=range_window_boundary,
)
