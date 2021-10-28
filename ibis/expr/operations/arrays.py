from public import public

from ...common import exceptions as com
from .. import datatypes as dt
from .. import rules as rlz
from ..signature import Argument as Arg
from .core import UnaryOp, ValueOp


@public
class ArrayColumn(ValueOp):
    cols = Arg(rlz.value_list_of(rlz.column(rlz.any), min_length=1))

    def _validate(self):
        if len({col.type() for col in self.cols}) > 1:
            raise com.IbisTypeError(
                f'The types of all input columns must match exactly in a '
                f'{type(self).__name__} operation.'
            )

    def output_type(self):
        first_dtype = self.cols[0].type()
        return dt.Array(first_dtype).column_type()


@public
class ArrayLength(UnaryOp):
    arg = Arg(rlz.array)
    output_type = rlz.shape_like('arg', dt.int64)


@public
class ArraySlice(ValueOp):
    arg = Arg(rlz.array)
    start = Arg(rlz.integer)
    stop = Arg(rlz.integer, default=None)
    output_type = rlz.typeof('arg')


@public
class ArrayIndex(ValueOp):
    arg = Arg(rlz.array)
    index = Arg(rlz.integer)

    def output_type(self):
        value_dtype = self.arg.type().value_type
        return rlz.shape_like(self.arg, value_dtype)


@public
class ArrayConcat(ValueOp):
    left = Arg(rlz.array)
    right = Arg(rlz.array)
    output_type = rlz.shape_like('left')

    def _validate(self):
        left_dtype, right_dtype = self.left.type(), self.right.type()
        if left_dtype != right_dtype:
            raise com.IbisTypeError(
                'Array types must match exactly in a {} operation. '
                'Left type {} != Right type {}'.format(
                    type(self).__name__, left_dtype, right_dtype
                )
            )


@public
class ArrayRepeat(ValueOp):
    arg = Arg(rlz.array)
    times = Arg(rlz.integer)
    output_type = rlz.typeof('arg')
