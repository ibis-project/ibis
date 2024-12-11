"""Reduction operations."""

from __future__ import annotations

from typing import Literal, Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import ValidationError, attribute
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.core import Column, Value
from ibis.expr.operations.relations import Relation  # noqa: TC001
from ibis.expr.operations.sortkeys import SortKey  # noqa: TC001


@public
class Reduction(Value):
    """Base class for reduction operations."""

    shape = ds.scalar


# TODO(kszucs): all reductions all filterable so we could remove Filterable
class Filterable(Value):
    where: Optional[Value[dt.Boolean]] = None


@public
class Count(Filterable, Reduction):
    """Count the number of non-null elements of a column."""

    arg: Column[dt.Any]

    dtype = dt.int64


@public
class CountStar(Filterable, Reduction):
    """Count the number of rows of a relation."""

    arg: Relation

    dtype = dt.int64

    @attribute
    def relations(self):
        return frozenset({self.arg})


@public
class CountDistinctStar(Filterable, Reduction):
    """Count the number of distinct rows of a relation."""

    arg: Relation

    dtype = dt.int64

    @attribute
    def relations(self):
        return frozenset({self.arg})


@public
class Arbitrary(Filterable, Reduction):
    """Retrieve an arbitrary element.

    Returns a non-null value unless the column is empty or all values are NULL.
    """

    arg: Column[dt.Any]

    dtype = rlz.dtype_like("arg")


@public
class First(Filterable, Reduction):
    """Retrieve the first element."""

    arg: Column[dt.Any]
    order_by: VarTuple[SortKey] = ()
    include_null: bool = False

    dtype = rlz.dtype_like("arg")


@public
class Last(Filterable, Reduction):
    """Retrieve the last element."""

    arg: Column[dt.Any]
    order_by: VarTuple[SortKey] = ()
    include_null: bool = False

    dtype = rlz.dtype_like("arg")


@public
class BitAnd(Filterable, Reduction):
    """Aggregate bitwise AND operation.

    All elements in an integer column are ANDed together.

    This can be used to determine which bit flags are set on all elements.

    See Also
    --------
    * BigQuery [`BIT_AND`](https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_and)
    * MySQL [`BIT_AND`](https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-and)
    """

    arg: Column[dt.Integer]

    dtype = rlz.dtype_like("arg")


@public
class BitOr(Filterable, Reduction):
    """Aggregate bitwise OR operation.

    All elements in an integer column are ORed together. This can be used
    to determine which bit flags are set on any element.

    See Also
    --------
    * BigQuery [`BIT_OR`](https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_or)
    * MySQL [`BIT_OR`](https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-or)
    """

    arg: Column[dt.Integer]

    dtype = rlz.dtype_like("arg")


@public
class BitXor(Filterable, Reduction):
    """Aggregate bitwise XOR operation.

    All elements in an integer column are XORed together. This can be used
    as a parity checksum of element values.

    See Also
    --------
    * BigQuery [`BIT_XOR`](https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_xor)
    * MySQL [`BIT_XOR`](https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-xor)
    """

    arg: Column[dt.Integer]

    dtype = rlz.dtype_like("arg")


@public
class Sum(Filterable, Reduction):
    """Compute the sum of a column."""

    arg: Column[dt.Numeric | dt.Boolean]

    @attribute
    def dtype(self):
        dtype = self.arg.dtype
        if dtype.is_boolean():
            return dt.int64
        elif dtype.is_integer():
            return dt.int64
        elif dtype.is_unsigned_integer():
            return dt.uint64
        elif dtype.is_floating():
            return dt.float64
        elif dtype.is_decimal():
            return dt.Decimal(
                precision=max(dtype.precision, 38)
                if dtype.precision is not None
                else None,
                scale=max(dtype.scale, 2) if dtype.scale is not None else None,
            )
        else:
            raise TypeError(f"Cannot compute sum of {dtype} values")


@public
class Mean(Filterable, Reduction):
    """Compute the mean of a column."""

    arg: Column[dt.Numeric | dt.Boolean]

    @attribute
    def dtype(self):
        if (dtype := self.arg.dtype).is_boolean():
            return dt.float64
        else:
            return dt.higher_precedence(dtype, dt.float64)


class QuantileBase(Filterable, Reduction):
    arg: Column

    @attribute
    def dtype(self):
        dtype = self.arg.dtype
        if dtype.is_numeric():
            dtype = dt.higher_precedence(dtype, dt.float64)
        return dtype


@public
class Median(QuantileBase):
    """Compute the median of a column."""


@public
class ApproxMedian(Median):
    """Compute the approximate median of a column."""


@public
class Quantile(QuantileBase):
    """Compute the quantile of a column."""

    quantile: Value[dt.Numeric]


@public
class ApproxQuantile(Quantile):
    """Compute the approximate quantile of a column."""

    arg: Column[dt.Numeric]


@public
class MultiQuantile(Filterable, Reduction):
    """Compute multiple quantiles of a column."""

    arg: Column
    quantile: Value[dt.Array[dt.Numeric]]

    @attribute
    def dtype(self):
        dtype = self.arg.dtype
        if dtype.is_numeric():
            dtype = dt.higher_precedence(dtype, dt.float64)
        return dt.Array(dtype)


@public
class ApproxMultiQuantile(MultiQuantile):
    """Compute multiple approximate quantiles of a column."""

    arg: Column[dt.Numeric]


class VarianceBase(Filterable, Reduction):
    """Base class for variance and standard deviation."""

    arg: Column[dt.Numeric | dt.Boolean]
    how: Literal["sample", "pop"]

    @attribute
    def dtype(self):
        if self.arg.dtype.is_decimal():
            return self.arg.dtype
        else:
            return dt.float64


@public
class StandardDev(VarianceBase):
    """Compute the standard deviation of a column."""


@public
class Variance(VarianceBase):
    """Compute the variance of a column."""


@public
class Correlation(Filterable, Reduction):
    """Correlation coefficient of two columns."""

    left: Column[dt.Numeric | dt.Boolean]
    right: Column[dt.Numeric | dt.Boolean]
    how: Literal["sample", "pop"] = "sample"

    dtype = dt.float64


@public
class Covariance(Filterable, Reduction):
    """Covariance of two columns."""

    left: Column[dt.Numeric | dt.Boolean]
    right: Column[dt.Numeric | dt.Boolean]
    how: Literal["sample", "pop"]

    dtype = dt.float64


@public
class Mode(Filterable, Reduction):
    """Compute the mode of a column."""

    arg: Column

    dtype = rlz.dtype_like("arg")


@public
class Max(Filterable, Reduction):
    """Compute the maximum of a column."""

    arg: Column

    dtype = rlz.dtype_like("arg")


@public
class Min(Filterable, Reduction):
    """Compute the minimum of a column."""

    arg: Column

    dtype = rlz.dtype_like("arg")


@public
class ArgMax(Filterable, Reduction):
    """Compute the index of the maximum value in a column."""

    arg: Column
    key: Column

    dtype = rlz.dtype_like("arg")


@public
class ArgMin(Filterable, Reduction):
    """Compute the index of the minimum value in a column."""

    arg: Column
    key: Column

    dtype = rlz.dtype_like("arg")


@public
class GroupConcat(Filterable, Reduction):
    """Concatenate strings in a group with a given separator character."""

    arg: Column
    sep: Value[dt.String]
    order_by: VarTuple[SortKey] = ()

    dtype = dt.string


@public
class CountDistinct(Filterable, Reduction):
    """Count the number of distinct values in a column."""

    arg: Column

    dtype = dt.int64


@public
class ApproxCountDistinct(CountDistinct):
    """Approximate number of unique values."""


@public
class ArrayCollect(Filterable, Reduction):
    """Collect values into an array."""

    arg: Column
    order_by: VarTuple[SortKey] = ()
    include_null: bool = False
    distinct: bool = False

    def __init__(self, arg, order_by, distinct, **kwargs):
        if distinct and order_by and [arg] != [key.expr for key in order_by]:
            raise ValidationError(
                "`collect` with `order_by` and `distinct=True` and may only "
                "order by the collected column"
            )
        super().__init__(arg=arg, order_by=order_by, distinct=distinct, **kwargs)

    @attribute
    def dtype(self):
        return dt.Array(self.arg.dtype)


@public
class All(Filterable, Reduction):
    """Check if all values in a column are true."""

    arg: Column[dt.Boolean]

    dtype = dt.boolean


@public
class Any(Filterable, Reduction):
    """Check if any value in a column is true."""

    arg: Column[dt.Boolean]

    dtype = dt.boolean
