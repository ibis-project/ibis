from __future__ import annotations

from itertools import product, starmap
from typing import Optional

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.common.annotations import attribute
from ibis.common.grounds import Concrete
from ibis.common.patterns import CoercionError, NoMatch, Pattern
from ibis.common.temporal import IntervalUnit


@public
def highest_precedence_shape(nodes):
    return max(node.shape for node in nodes)


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
    return dt.highest_precedence(node.dtype for node in nodes)


@public
def castable(source, target):
    """Return whether source ir type is implicitly castable to target.

    Based on the underlying datatypes and the value in case of Literals
    """
    value = getattr(source, "value", None)
    return source.dtype.castable(target.dtype, value=value)


@public
def comparable(left, right):
    return castable(left, right) or castable(right, left)


# ---------------------------------------------------------------------
# Output type functions


@public
def dtype_like(name):
    @attribute
    def dtype(self):
        args = getattr(self, name)
        args = args if util.is_iterable(args) else [args]
        return highest_precedence_dtype(args)

    return dtype


@public
def shape_like(name):
    @attribute
    def shape(self):
        args = getattr(self, name)
        args = args if util.is_iterable(args) else [args]
        args = [a for a in args if a is not None]
        return highest_precedence_shape(args)

    return shape


# TODO(kszucs): might just use bounds instead of actual literal values
# that could simplify interval binop output_type methods
# TODO(kszucs): pre-generate mapping?


def _promote_integral_binop(exprs, op):
    bounds, dtypes = [], []
    for arg in exprs:
        dtypes.append(arg.dtype)
        if isinstance(arg, ops.Literal) and arg.value is not None:
            bounds.append([arg.value])
        else:
            bounds.append(arg.dtype.bounds)

    all_unsigned = dtypes and util.all_of(dtypes, dt.UnsignedInteger)
    # In some cases, the bounding type might be int8, even though neither
    # of the types are that small. We want to ensure the containing type is
    # _at least_ as large as the smallest type in the expression.
    values = starmap(op, product(*bounds))
    dtypes += [dt.infer(v, prefer_unsigned=all_unsigned) for v in values]

    return dt.highest_precedence(dtypes)


@public
def numeric_like(name, op):
    @attribute
    def dtype(self):
        args = getattr(self, name)
        dtypes = [arg.dtype for arg in args]
        if util.all_of(dtypes, dt.Integer):
            result = _promote_integral_binop(args, op)
        else:
            result = highest_precedence_dtype(args)

        return result

    return dtype


def _promote_interval_resolution(units: list[IntervalUnit]) -> IntervalUnit:
    # Find the smallest unit present in units
    for unit in reversed(IntervalUnit):
        if unit in units:
            return unit
    raise AssertionError("unreachable")


def arg_type_error_format(op: ops.Value) -> str:
    if isinstance(op, ops.Literal):
        return f"Literal({op.value}):{op.dtype}"
    else:
        return f"{op.name}:{op.dtype}"


class ValueOf(Concrete, Pattern):
    """Match a value of a specific type **instance**.

    This is different from the Value[T] annotations which construct
    GenericCoercedTo(Value[T]) validators working with datatype types
    rather than instances.

    Parameters
    ----------
    dtype : DataType | None
        The datatype the constructed Value instance should conform to.

    """

    dtype: Optional[dt.DataType] = None

    def match(self, value, context):
        try:
            value = ops.Value.__coerce__(value, self.dtype)
        except CoercionError:
            return NoMatch

        if self.dtype and not value.dtype.castable(self.dtype):
            return NoMatch

        return value
