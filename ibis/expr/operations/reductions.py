from __future__ import annotations

from typing import Literal, Optional

from public import public

import ibis.common.exceptions as exc
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Column, Value
from ibis.expr.operations.relations import Relation  # noqa: TCH001


@public
class Reduction(Value):
    shape = ds.scalar

    @property
    def __window_op__(self):
        return self


class Filterable(Value):
    where: Optional[Value[dt.Boolean]] = None


@public
class Count(Filterable, Reduction):
    arg: Column[dt.Any]

    dtype = dt.int64


@public
class CountStar(Filterable, Reduction):
    arg: Relation

    dtype = dt.int64


@public
class CountDistinctStar(Filterable, Reduction):
    arg: Relation

    dtype = dt.int64


@public
class Arbitrary(Filterable, Reduction):
    arg: Column[dt.Any]
    how: Literal["first", "last", "heavy"]

    dtype = rlz.dtype_like("arg")


@public
class First(Filterable, Reduction):
    """Retrieve the first element."""

    arg: Column[dt.Any]

    dtype = rlz.dtype_like("arg")

    @property
    def __window_op__(self):
        import ibis.expr.operations as ops

        if self.where is not None:
            raise exc.OperationNotDefinedError(
                "FirstValue cannot be filtered in a window context"
            )
        return ops.FirstValue(arg=self.arg)


@public
class Last(Filterable, Reduction):
    """Retrieve the last element."""

    arg: Column[dt.Any]

    dtype = rlz.dtype_like("arg")

    @property
    def __window_op__(self):
        import ibis.expr.operations as ops

        if self.where is not None:
            raise exc.OperationNotDefinedError(
                "LastValue cannot be filtered in a window context"
            )
        return ops.LastValue(arg=self.arg)


@public
class BitAnd(Filterable, Reduction):
    """Aggregate bitwise AND operation.

    All elements in an integer column are ANDed together.

    This can be used to determine which bit flags are set on all elements.

    Resources:

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

    Resources:

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

    Resources:

    * BigQuery [`BIT_XOR`](https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_xor)
    * MySQL [`BIT_XOR`](https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-xor)
    """

    arg: Column[dt.Integer]

    dtype = rlz.dtype_like("arg")


@public
class Sum(Filterable, Reduction):
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
    arg: Column[dt.Numeric | dt.Boolean]

    @attribute
    def dtype(self):
        if (dtype := self.arg.dtype).is_boolean():
            return dt.float64
        else:
            return dt.higher_precedence(dtype, dt.float64)


@public
class Median(Filterable, Reduction):
    arg: Column[dt.Numeric | dt.Boolean]

    @attribute
    def dtype(self):
        return dt.higher_precedence(self.arg.dtype, dt.float64)


@public
class Quantile(Filterable, Reduction):
    arg: Value
    quantile: Value[dt.Numeric]

    dtype = dt.float64


@public
class MultiQuantile(Filterable, Reduction):
    arg: Value
    quantile: Value[dt.Array[dt.Float64]]

    dtype = dt.Array(dt.float64)


@public
class VarianceBase(Filterable, Reduction):
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
    pass


@public
class Variance(VarianceBase):
    pass


@public
class Correlation(Filterable, Reduction):
    """Coefficient of correlation of a set of number pairs."""

    left: Column[dt.Numeric | dt.Boolean]
    right: Column[dt.Numeric | dt.Boolean]
    how: Literal["sample", "pop"] = "sample"

    dtype = dt.float64


@public
class Covariance(Filterable, Reduction):
    """Covariance of a set of number pairs."""

    left: Column[dt.Numeric | dt.Boolean]
    right: Column[dt.Numeric | dt.Boolean]
    how: Literal["sample", "pop"]

    dtype = dt.float64


@public
class Mode(Filterable, Reduction):
    arg: Column

    dtype = rlz.dtype_like("arg")


@public
class Max(Filterable, Reduction):
    arg: Column

    dtype = rlz.dtype_like("arg")


@public
class Min(Filterable, Reduction):
    arg: Column

    dtype = rlz.dtype_like("arg")


@public
class ArgMax(Filterable, Reduction):
    arg: Column
    key: Column

    dtype = rlz.dtype_like("arg")


@public
class ArgMin(Filterable, Reduction):
    arg: Column
    key: Column

    dtype = rlz.dtype_like("arg")


@public
class ApproxCountDistinct(Filterable, Reduction):
    """Approximate number of unique values using HyperLogLog algorithm.

    Impala offers the NDV built-in function for this.
    """

    arg: Column

    # Impala 2.0 and higher returns a DOUBLE
    dtype = dt.int64


@public
class ApproxMedian(Filterable, Reduction):
    """Compute the approximate median of a set of comparable values."""

    arg: Column

    dtype = rlz.dtype_like("arg")


@public
class GroupConcat(Filterable, Reduction):
    arg: Column
    sep: Value[dt.String]

    dtype = dt.string


@public
class CountDistinct(Filterable, Reduction):
    arg: Column

    dtype = dt.int64


@public
class ArrayCollect(Filterable, Reduction):
    arg: Column

    @attribute
    def dtype(self):
        return dt.Array(self.arg.dtype)


@public
class All(Filterable, Reduction):
    arg: Column[dt.Boolean]

    dtype = dt.boolean


@public
class Any(Filterable, Reduction):
    arg: Column[dt.Boolean]

    dtype = dt.boolean
