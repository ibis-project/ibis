from public import public

from ibis.common import exceptions as com
from ibis.common.validators import immutable_property
from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import UnaryOp, ValueOp


@public
class ArrayColumn(ValueOp):
    cols = rlz.value_list_of(rlz.column(rlz.any), min_length=1)

    output_shape = rlz.Shape.COLUMNAR

    def __init__(self, cols):
        if len({col.type() for col in cols}) > 1:
            raise com.IbisTypeError(
                f'The types of all input columns must match exactly in a '
                f'{type(self).__name__} operation.'
            )
        super().__init__(cols=cols)

    @immutable_property
    def output_dtype(self):
        first_dtype = self.cols[0].type()
        return dt.Array(first_dtype)


@public
class ArrayLength(UnaryOp):
    arg = rlz.array

    output_dtype = dt.int64
    output_shape = rlz.shape_like("args")


@public
class ArraySlice(ValueOp):
    arg = rlz.array
    start = rlz.integer
    stop = rlz.optional(rlz.integer)

    output_dtype = rlz.dtype_like("arg")
    output_shape = rlz.shape_like("arg")


@public
class ArrayIndex(ValueOp):
    arg = rlz.array
    index = rlz.integer

    output_shape = rlz.shape_like("args")

    @immutable_property
    def output_dtype(self):
        return self.arg.type().value_type


@public
class ArrayConcat(ValueOp):
    left = rlz.array
    right = rlz.array

    output_dtype = rlz.dtype_like("left")
    output_shape = rlz.shape_like("args")

    def __init__(self, left, right):
        left_dtype, right_dtype = left.type(), right.type()
        if left_dtype != right_dtype:
            raise com.IbisTypeError(
                'Array types must match exactly in a {} operation. '
                'Left type {} != Right type {}'.format(
                    type(self).__name__, left_dtype, right_dtype
                )
            )
        super().__init__(left=left, right=right)


@public
class ArrayRepeat(ValueOp):
    arg = rlz.array
    times = rlz.integer

    output_dtype = rlz.dtype_like("arg")
    output_shape = rlz.shape_like("args")
