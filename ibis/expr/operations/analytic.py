from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Column, Scalar, Value


@public
class Analytic(Value):
    output_shape = ds.columnar

    @property
    def __window_op__(self):
        return self


@public
class ShiftBase(Analytic):
    arg: Column[dt.Any]
    offset: Optional[Value[dt.Integer | dt.Interval]] = None
    default: Optional[Value] = None

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
    arg: Column[dt.Any]


@public
class DenseRank(RankBase):
    arg: Column[dt.Any]


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
class Cumulative(Analytic):
    pass


@public
class CumulativeSum(Cumulative):
    """Cumulative sum.

    Requires an ordering window.
    """

    arg: Column[dt.Numeric]

    @attribute.default
    def output_dtype(self):
        return dt.higher_precedence(self.arg.output_dtype.largest, dt.int64)


@public
class CumulativeMean(Cumulative):
    """Cumulative mean.

    Requires an order window.
    """

    arg: Column[dt.Numeric]

    @attribute.default
    def output_dtype(self):
        return dt.higher_precedence(self.arg.output_dtype.largest, dt.float64)


@public
class CumulativeMax(Cumulative):
    arg: Column[dt.Any]

    output_dtype = rlz.dtype_like("arg")


@public
class CumulativeMin(Cumulative):
    """Cumulative min.

    Requires an order window.
    """

    arg: Column[dt.Any]

    output_dtype = rlz.dtype_like("arg")


@public
class CumulativeAny(Cumulative):
    arg: Column[dt.Boolean]
    output_dtype = rlz.dtype_like("arg")


@public
class CumulativeAll(Cumulative):
    arg: Column[dt.Boolean]

    output_dtype = rlz.dtype_like("arg")


@public
class PercentRank(Analytic):
    arg: Column[dt.Any]

    output_dtype = dt.double


@public
class CumeDist(Analytic):
    arg: Column[dt.Any]

    output_dtype = dt.double


@public
class NTile(Analytic):
    arg: Column[dt.Any]
    buckets: Scalar[dt.Integer]

    output_dtype = dt.int64


@public
class FirstValue(Analytic):
    """Retrieve the first element."""

    arg: Column[dt.Any]

    output_dtype = rlz.dtype_like("arg")


@public
class LastValue(Analytic):
    """Retrieve the last element."""

    arg: Column[dt.Any]

    output_dtype = rlz.dtype_like("arg")


@public
class NthValue(Analytic):
    """Retrieve the Nth element."""

    arg: Column[dt.Any]
    nth: Value[dt.Integer]

    output_dtype = rlz.dtype_like("arg")


public(AnalyticOp=Analytic, CumulativeOp=Cumulative)
