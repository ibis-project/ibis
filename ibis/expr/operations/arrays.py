"""Operations for array expressions."""

from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.core import Argument, Unary, Value


@public
class Array(Value):
    """Construct an array."""

    exprs: VarTuple[Value]

    @attribute
    def shape(self):
        return rlz.highest_precedence_shape(self.exprs)

    @attribute
    def dtype(self):
        return dt.Array(rlz.highest_precedence_dtype(self.exprs))


@public
class ArrayLength(Unary):
    """Compute the length of an array."""

    arg: Value[dt.Array]

    dtype = dt.int64
    shape = rlz.shape_like("args")


@public
class ArraySlice(Value):
    """Slice an array element."""

    arg: Value[dt.Array]
    start: Value[dt.Integer]
    stop: Optional[Value[dt.Integer]] = None

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("args")


@public
class ArrayIndex(Value):
    """Return the element of an array at some index."""

    arg: Value[dt.Array]
    index: Value[dt.Integer]

    shape = rlz.shape_like("args")

    @attribute
    def dtype(self):
        return self.arg.dtype.value_type


@public
class ArrayConcat(Value):
    """Concatenate two or more arrays into a single array."""

    arg: VarTuple[Value[dt.Array]]

    shape = rlz.shape_like("arg")

    @attribute
    def dtype(self):
        return dt.Array(dt.highest_precedence(arg.dtype.value_type for arg in self.arg))


@public
class ArrayRepeat(Value):
    """Repeat the elements of an array."""

    arg: Value[dt.Array]
    times: Value[dt.Integer]

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("args")


@public
class ArrayMap(Value):
    """Apply a function to every element of an array."""

    arg: Value[dt.Array]
    body: Value

    param: Argument
    index: Argument | None

    shape = rlz.shape_like("arg")

    @attribute
    def dtype(self) -> dt.DataType:
        return dt.Array(self.body.dtype)


@public
class ArrayFilter(Value):
    """Filter array elements with a function."""

    arg: Value[dt.Array]
    body: Value[dt.Boolean]

    param: Argument
    index: Argument | None

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Unnest(Value):
    """Unnest an array value into a column."""

    arg: Value[dt.Array]

    shape = ds.columnar

    @attribute
    def dtype(self):
        return self.arg.dtype.value_type


@public
class ArrayContains(Value):
    """Return whether an array contains a specific value."""

    arg: Value[dt.Array]
    other: Value

    dtype = dt.boolean
    shape = rlz.shape_like("args")


@public
class ArrayPosition(Value):
    """Return the position of a specific value in an array."""

    arg: Value[dt.Array]
    other: Value

    dtype = dt.int64
    shape = rlz.shape_like("args")


@public
class ArrayRemove(Value):
    """Remove an element from an array."""

    arg: Value[dt.Array]
    other: Value

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("args")


@public
class ArrayDistinct(Value):
    """Return the unique elements of an array."""

    arg: Value[dt.Array]

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("arg")


@public
class ArraySort(Value):
    """Sort the values of an array."""

    arg: Value[dt.Array]

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("arg")


@public
class ArrayUnion(Value):
    """Return the union of two arrays."""

    left: Value[dt.Array]
    right: Value[dt.Array]

    dtype = rlz.dtype_like("args")
    shape = rlz.shape_like("args")


@public
class ArrayIntersect(Value):
    """Return the intersection of two arrays."""

    left: Value[dt.Array]
    right: Value[dt.Array]

    dtype = rlz.dtype_like("args")
    shape = rlz.shape_like("args")


@public
class ArrayZip(Value):
    """Zip two or more arrays into an array of structs."""

    arg: VarTuple[Value[dt.Array]]

    shape = rlz.shape_like("arg")

    @attribute
    def dtype(self):
        return dt.Array(
            dt.Struct(
                {
                    f"f{i:d}": array.dtype.value_type
                    for i, array in enumerate(self.arg, start=1)
                }
            )
        )


@public
class ArrayFlatten(Value):
    """Flatten a nested array one level.

    The input expression must have at least one level of nesting for flattening
    to make sense.
    """

    arg: Value[dt.Array[dt.Array]]
    shape = rlz.shape_like("arg")

    @property
    def dtype(self):
        return self.arg.dtype.value_type


class Range(Value):
    """Base class for range-generating operations."""

    shape = rlz.shape_like("args")

    @attribute
    def dtype(self) -> dt.DataType:
        return dt.Array(dt.highest_precedence((self.start.dtype, self.stop.dtype)))


@public
class IntegerRange(Range):
    """Produce an array of integers from `start` to `stop`, moving by `step`."""

    start: Value[dt.Integer]
    stop: Value[dt.Integer]
    step: Value[dt.Integer]


@public
class TimestampRange(Range):
    """Produce an array of timestamps from `start` to `stop`, moving by `step`."""

    start: Value[dt.Timestamp]
    stop: Value[dt.Timestamp]
    step: Value[dt.Interval]


@public
class ArrayAgg(Value):
    arg: Value[dt.Array]
    shape = rlz.shape_like("args")

    @attribute
    def dtype(self) -> dt.DataType:
        return self.arg.dtype.value_type


@public
class ArrayMin(ArrayAgg):
    """Compute the minimum value of an array."""


@public
class ArrayMax(ArrayAgg):
    """Compute the maximum value of an array."""


@public
class ArrayMode(ArrayAgg):
    """Compute the mode of an array."""


# in duckdb summing an array of ints leads to an int, but for other backends
# it might lead to a float??
@public
class ArraySum(ArrayAgg):
    """Compute the sum of an array."""

    arg: Value[dt.Array[dt.Numeric]]


@public
class ArrayMean(ArrayAgg):
    """Compute the average of an array."""

    arg: Value[dt.Array[dt.Numeric]]

    @attribute
    def dtype(self) -> dt.DataType:
        dtype = self.arg.dtype.value_type
        if dtype.is_floating() or dtype.is_integer():
            return dt.float64
        # do nothing for decimal types
        return dtype


@public
class ArrayAny(ArrayAgg):
    """Compute whether any array element is true."""

    arg: Value[dt.Array[dt.Boolean]]


@public
class ArrayAll(ArrayAgg):
    """Compute whether all array elements are true."""

    arg: Value[dt.Array[dt.Boolean]]
