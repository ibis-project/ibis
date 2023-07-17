from __future__ import annotations

from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Argument, Unary, Value


@public
class ArrayColumn(Value):
    cols = rlz.tuple_of(rlz.any, min_length=1)

    output_shape = rlz.Shape.COLUMNAR

    @attribute.default
    def output_dtype(self):
        return dt.Array(rlz.highest_precedence_dtype(self.cols))


@public
class ArrayLength(Unary):
    arg = rlz.array

    output_dtype = dt.int64
    output_shape = rlz.shape_like("args")


@public
class ArraySlice(Value):
    arg = rlz.array
    start = rlz.integer
    stop = rlz.optional(rlz.integer)

    output_dtype = rlz.dtype_like("arg")
    output_shape = rlz.shape_like("arg")


@public
class ArrayIndex(Value):
    arg = rlz.array
    index = rlz.integer

    output_shape = rlz.shape_like("args")

    @attribute.default
    def output_dtype(self):
        return self.arg.output_dtype.value_type


@public
class ArrayConcat(Value):
    left = rlz.array
    right = rlz.array

    output_dtype = rlz.dtype_like("left")
    output_shape = rlz.shape_like("args")

    def __init__(self, left, right):
        if left.output_dtype != right.output_dtype:
            raise com.IbisTypeError(
                'Array types must match exactly in a {} operation. '
                'Left type {} != Right type {}'.format(
                    type(self).__name__, left.output_dtype, right.output_dtype
                )
            )
        super().__init__(left=left, right=right)


@public
class ArrayRepeat(Value):
    arg = rlz.array
    times = rlz.integer

    output_dtype = rlz.dtype_like("arg")
    output_shape = rlz.shape_like("args")


class ArrayApply(Value):
    arg = rlz.array

    @attribute.default
    def parameter(self):
        (name,) = self.func.__signature__.parameters.keys()
        return name

    @attribute.default
    def result(self):
        arg = Argument(
            name=self.parameter,
            shape=self.arg.output_shape,
            dtype=self.arg.output_dtype.value_type,
        )
        return self.func(arg)

    @attribute.default
    def output_shape(self):
        return self.arg.output_shape


@public
class ArrayMap(ArrayApply):
    func = rlz.callable_with([rlz.expr_of(rlz.any)], rlz.any)

    @attribute.default
    def output_dtype(self) -> dt.DataType:
        return dt.Array(self.result.output_dtype)


@public
class ArrayFilter(ArrayApply):
    func = rlz.callable_with([rlz.expr_of(rlz.any)], rlz.boolean)

    output_dtype = rlz.dtype_like("arg")


@public
class Unnest(Value):
    arg = rlz.array

    @attribute.default
    def output_dtype(self):
        return self.arg.output_dtype.value_type

    output_shape = rlz.Shape.COLUMNAR


@public
class ArrayContains(Value):
    arg = rlz.array
    other = rlz.any

    output_dtype = dt.boolean
    output_shape = rlz.shape_like("args")


@public
class ArrayPosition(Value):
    arg = rlz.array
    other = rlz.any

    output_dtype = dt.int64
    output_shape = rlz.shape_like("args")


@public
class ArrayRemove(Value):
    arg = rlz.array
    other = rlz.any

    output_dtype = rlz.dtype_like("arg")
    output_shape = rlz.shape_like("args")


@public
class ArrayDistinct(Value):
    arg = rlz.array
    output_dtype = rlz.dtype_like("arg")
    output_shape = rlz.shape_like("arg")


@public
class ArraySort(Value):
    arg = rlz.array

    output_dtype = rlz.dtype_like("arg")
    output_shape = rlz.shape_like("arg")


@public
class ArrayUnion(Value):
    left = rlz.array
    right = rlz.array

    output_dtype = rlz.dtype_like("args")
    output_shape = rlz.shape_like("args")


@public
class ArrayZip(Value):
    arg = rlz.tuple_of(rlz.array, min_length=2)

    output_shape = rlz.shape_like("arg")

    @attribute.default
    def output_dtype(self):
        return dt.Array(
            dt.Struct(
                {
                    f"f{i:d}": array.output_dtype.value_type
                    for i, array in enumerate(self.arg, start=1)
                }
            )
        )
