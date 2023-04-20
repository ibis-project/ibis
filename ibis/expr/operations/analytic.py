from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Value


@public
class Analytic(Value):
    output_shape = rlz.Shape.COLUMNAR

    @property
    def __window_op__(self):
        return self


@public
class ShiftBase(Analytic):
    arg = rlz.column(rlz.any)

    offset = rlz.optional(rlz.one_of((rlz.integer, rlz.interval)))
    default = rlz.optional(rlz.any)

    output_dtype = rlz.dtype_like("arg")


@public
class Lag(ShiftBase):
    pass


@public
class Lead(ShiftBase):
    pass


@public
class RankBase(Analytic):
    output_dtype = dt.int64


@public
class MinRank(RankBase):
    arg = rlz.column(rlz.any)


@public
class DenseRank(RankBase):
    arg = rlz.column(rlz.any)


@public
class RowNumber(RankBase):
    """Compute the row number over a window, starting from 0.

    Equivalent to SQL's `ROW_NUMBER()`.

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('values', dt.int64)])
    >>> w = ibis.window(order_by=t.values)
    >>> row_num = ibis.row_number().over(w)
    >>> result = t[t.values, row_num.name('row_num')]

    Returns
    -------
    IntegerColumn
        Row number
    """


@public
class CumulativeOp(Analytic):
    pass


@public
class CumulativeSum(CumulativeOp):
    """Cumulative sum.

    Requires an ordering window.
    """

    arg = rlz.column(rlz.numeric)

    @attribute.default
    def output_dtype(self):
        return dt.higher_precedence(self.arg.output_dtype.largest, dt.int64)


@public
class CumulativeMean(CumulativeOp):
    """Cumulative mean.

    Requires an order window.
    """

    arg = rlz.column(rlz.numeric)

    @attribute.default
    def output_dtype(self):
        return dt.higher_precedence(self.arg.output_dtype.largest, dt.float64)


@public
class CumulativeMax(CumulativeOp):
    arg = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like("arg")


@public
class CumulativeMin(CumulativeOp):
    """Cumulative min.

    Requires an order window.
    """

    arg = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like("arg")


@public
class CumulativeAny(CumulativeOp):
    arg = rlz.column(rlz.boolean)
    output_dtype = rlz.dtype_like("arg")


@public
class CumulativeAll(CumulativeOp):
    arg = rlz.column(rlz.boolean)
    output_dtype = rlz.dtype_like("arg")


@public
class PercentRank(Analytic):
    arg = rlz.column(rlz.any)
    output_dtype = dt.double


@public
class CumeDist(Analytic):
    arg = rlz.column(rlz.any)
    output_dtype = dt.double


@public
class NTile(Analytic):
    arg = rlz.column(rlz.any)
    buckets = rlz.scalar(rlz.integer)
    output_dtype = dt.int64


@public
class FirstValue(Analytic):
    """Retrieve the first element."""

    arg = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like("arg")


@public
class LastValue(Analytic):
    """Retrieve the last element."""

    arg = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like("arg")


@public
class NthValue(Analytic):
    """Retrieve the Nth element."""

    arg = rlz.column(rlz.any)
    nth = rlz.integer
    output_dtype = rlz.dtype_like("arg")


public(AnalyticOp=Analytic)
