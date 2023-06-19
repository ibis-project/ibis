from __future__ import annotations

from typing import Callable, Optional

from public import public

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.operations.core import Argument, Unary, Value


@public
class ArrayColumn(Value):
    cols: VarTuple[Value]

    shape = ds.columnar

    @attribute.default
    def dtype(self):
        return dt.Array(rlz.highest_precedence_dtype(self.cols))


@public
class ArrayLength(Unary):
    arg: Value[dt.Array]

    dtype = dt.int64
    shape = rlz.shape_like("args")


@public
class ArraySlice(Value):
    arg: Value[dt.Array]
    start: Value[dt.Integer]
    stop: Optional[Value[dt.Integer]] = None

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("arg")


@public
class ArrayIndex(Value):
    arg: Value[dt.Array]
    index: Value[dt.Integer]

    shape = rlz.shape_like("args")

    @attribute.default
    def dtype(self):
        return self.arg.dtype.value_type


@public
class ArrayConcat(Value):
    left: Value[dt.Array]
    right: Value[dt.Array]

    dtype = rlz.dtype_like("left")
    shape = rlz.shape_like("args")

    def __init__(self, left, right):
        if left.dtype != right.dtype:
            raise com.IbisTypeError(
                'Array types must match exactly in a {} operation. '
                'Left type {} != Right type {}'.format(
                    type(self).__name__, left.dtype, right.dtype
                )
            )
        super().__init__(left=left, right=right)


@public
class ArrayRepeat(Value):
    arg: Value[dt.Array]
    times: Value[dt.Integer]

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("args")


class ArrayApply(Value):
    arg: Value[dt.Array]

    @attribute.default
    def parameter(self):
        (name,) = self.func.__signature__.parameters.keys()
        return name

    @attribute.default
    def result(self):
        arg = Argument(
            name=self.parameter,
            shape=self.arg.shape,
            dtype=self.arg.dtype.value_type,
        )
        return self.func(arg)

    @attribute.default
    def shape(self):
        return self.arg.shape


@public
class ArrayMap(ArrayApply):
    func: Callable[[Value], Value]

    @attribute.default
    def dtype(self) -> dt.DataType:
        return dt.Array(self.result.dtype)


@public
class ArrayFilter(ArrayApply):
    func: Callable[[Value], Value[dt.Boolean]]

    dtype = rlz.dtype_like("arg")


@public
class Unnest(Value):
    arg: Value[dt.Array]

    shape = ds.columnar

    @attribute.default
    def dtype(self):
        return self.arg.dtype.value_type


@public
class ArrayContains(Value):
    arg: Value[dt.Array]
    other: Value

    dtype = dt.boolean
    shape = rlz.shape_like("args")


@public
class ArrayPosition(Value):
    arg: Value[dt.Array]
    other: Value

    dtype = dt.int64
    shape = rlz.shape_like("args")


@public
class ArrayRemove(Value):
    arg: Value[dt.Array]
    other: Value

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("args")


@public
class ArrayDistinct(Value):
    arg: Value[dt.Array]

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("arg")


@public
class ArraySort(Value):
    arg: Value[dt.Array]

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("arg")


@public
class ArrayUnion(Value):
    left: Value[dt.Array]
    right: Value[dt.Array]

    dtype = rlz.dtype_like("args")
    shape = rlz.shape_like("args")


@public
class ArrayZip(Value):
    arg: VarTuple[Value[dt.Array]]

    shape = rlz.shape_like("arg")

    @attribute.default
    def dtype(self):
        return dt.Array(
            dt.Struct(
                {
                    f"f{i:d}": array.dtype.value_type
                    for i, array in enumerate(self.arg, start=1)
                }
            )
        )
