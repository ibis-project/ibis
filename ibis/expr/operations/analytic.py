from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations.core import Column, Scalar, Value


@public
class Analytic(Value):
    shape = ds.columnar

    @property
    def __window_op__(self):
        return self


@public
class ShiftBase(Analytic):
    arg: Column[dt.Any]
    offset: Optional[Value[dt.Integer | dt.Interval]] = None
    default: Optional[Value] = None

    dtype = rlz.dtype_like("arg")


@public
class Lag(ShiftBase):
    pass


@public
class Lead(ShiftBase):
    pass


@public
class RankBase(Analytic):
    dtype = dt.int64


@public
class MinRank(RankBase):
    pass


@public
class DenseRank(RankBase):
    pass


@public
class RowNumber(RankBase):
    """Compute the row number over a window, starting from 0.

    Equivalent to SQL's `ROW_NUMBER()`.

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([("values", dt.int64)])
    >>> w = ibis.window(order_by=t.values)
    >>> row_num = ibis.row_number().over(w)
    >>> result = t[t.values, row_num.name("row_num")]

    Returns
    -------
    IntegerColumn
        Row number
    """


@public
class PercentRank(Analytic):
    dtype = dt.double


@public
class CumeDist(Analytic):
    dtype = dt.double


@public
class NTile(Analytic):
    buckets: Scalar[dt.Integer]

    dtype = dt.int64


@public
class FirstValue(Analytic):
    """Retrieve the first element."""

    arg: Column[dt.Any]

    dtype = rlz.dtype_like("arg")


@public
class LastValue(Analytic):
    """Retrieve the last element."""

    arg: Column[dt.Any]

    dtype = rlz.dtype_like("arg")


@public
class NthValue(Analytic):
    """Retrieve the Nth element."""

    arg: Column[dt.Any]
    nth: Value[dt.Integer]

    dtype = rlz.dtype_like("arg")


public(AnalyticOp=Analytic)
