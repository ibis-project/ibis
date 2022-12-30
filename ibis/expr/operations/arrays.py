from __future__ import annotations

from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Unary, Value


@public
class ArrayColumn(Value):
    cols = rlz.tuple_of(rlz.column(rlz.any), min_length=1)

    output_shape = rlz.Shape.COLUMNAR

    def __init__(self, cols):
        unique_dtypes = {col.output_dtype for col in cols}
        if len(unique_dtypes) > 1:
            raise com.IbisTypeError(
                f'The types of all input columns must match exactly in a '
                f'{type(self).__name__} operation.'
            )
        super().__init__(cols=cols)

    @attribute.default
    def output_dtype(self):
        first_dtype = self.cols[0].output_dtype
        return dt.Array(first_dtype)


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


@public
class Unnest(Value):
    arg = rlz.array

    @attribute.default
    def output_dtype(self):
        return self.arg.output_dtype.value_type

    output_shape = rlz.Shape.COLUMNAR
