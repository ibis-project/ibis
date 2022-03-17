from public import public

from ...common import exceptions as com
from .. import datatypes as dt
from .. import rules as rlz
from .core import UnaryOp, ValueOp


@public
class ArrayColumn(ValueOp):
    cols = rlz.value_list_of(rlz.column(rlz.any), min_length=1)

    def __init__(self, cols):
        if len({col.type() for col in cols}) > 1:
            raise com.IbisTypeError(
                f'The types of all input columns must match exactly in a '
                f'{type(self).__name__} operation.'
            )
        super().__init__(cols=cols)

    def output_type(self):
        first_dtype = self.cols[0].type()
        return dt.Array(first_dtype).column_type()


@public
class ArrayLength(UnaryOp):
    arg = rlz.array
    output_type = rlz.shape_like('arg', dt.int64)


@public
class ArraySlice(ValueOp):
    arg = rlz.array
    start = rlz.integer
    stop = rlz.optional(rlz.integer)
    output_type = rlz.typeof('arg')


@public
class ArrayIndex(ValueOp):
    arg = rlz.array
    index = rlz.integer

    def output_type(self):
        value_dtype = self.arg.type().value_type
        return rlz.shape_like(self.arg, value_dtype)


@public
class ArrayConcat(ValueOp):
    left = rlz.array
    right = rlz.array
    output_type = rlz.shape_like('left')

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
    output_type = rlz.typeof('arg')
