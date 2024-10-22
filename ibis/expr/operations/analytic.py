"""Operations for analytic window functions."""

from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations.core import Column, Scalar, Value


@public
class Analytic(Value):
    """Base class for analytic window function operations."""

    shape = ds.columnar


class ShiftBase(Analytic):
    """Base class for shift operations."""

    arg: Column[dt.Any]
    offset: Optional[Value[dt.Integer | dt.Interval]] = None
    default: Optional[Value] = None

    dtype = rlz.dtype_like("arg")


@public
class Lag(ShiftBase):
    """Shift a column forward."""


@public
class Lead(ShiftBase):
    """Shift a column backward."""


@public
class RankBase(Analytic):
    """Base class for ranking operations."""

    dtype = dt.int64


@public
class MinRank(RankBase):
    """Rank within an ordered partition."""


@public
class DenseRank(RankBase):
    """Rank within an ordered partition, consecutively."""


@public
class RowNumber(RankBase):
    """Compute the row number over a window, starting from 0."""


@public
class PercentRank(Analytic):
    """Compute the percentile rank over a window."""

    dtype = dt.double


@public
class CumeDist(Analytic):
    """Compute the cumulative distribution function of a column over a window."""

    dtype = dt.double


@public
class NTile(Analytic):
    """Compute the percentile of a column over a window."""

    buckets: Scalar[dt.Integer]

    dtype = dt.int64


@public
class NthValue(Analytic):
    """Retrieve the Nth element of a column over a window."""

    arg: Column[dt.Any]
    nth: Value[dt.Integer]

    dtype = rlz.dtype_like("arg")


public(AnalyticOp=Analytic)
