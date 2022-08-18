from __future__ import annotations

from public import public

from ibis.common.validators import immutable_property
from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr import types as ir
from ibis.expr.operations.core import Value, distinct_roots
from ibis.expr.operations.generic import _Negatable
from ibis.util import deprecated


@public
class Reduction(Value):
    output_shape = rlz.Shape.SCALAR


class Filterable(Value):
    where = rlz.optional(rlz.boolean)


@public
class Count(Filterable, Reduction):
    arg = rlz.one_of((rlz.column(rlz.any), rlz.table))
    output_dtype = dt.int64


@public
class Arbitrary(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    how = rlz.optional(rlz.isin({'first', 'last', 'heavy'}))
    output_dtype = rlz.dtype_like('arg')


@public
class BitAnd(Filterable, Reduction):
    """Aggregate bitwise AND operation.

    All elements in an integer column are ANDed together.

    This can be used to determine which bit flags are set on all elements.

    Resources:

    * BigQuery [`BIT_AND`](https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_and)
    * MySQL [`BIT_AND`](https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-and)
    """  # noqa: E501

    arg = rlz.column(rlz.integer)
    output_dtype = rlz.dtype_like('arg')


@public
class BitOr(Filterable, Reduction):
    """Aggregate bitwise OR operation.

    All elements in an integer column are ORed together. This can be used
    to determine which bit flags are set on any element.

    Resources:

    * BigQuery [`BIT_OR`](https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_or)
    * MySQL [`BIT_OR`](https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-or)
    """  # noqa: E501

    arg = rlz.column(rlz.integer)
    output_dtype = rlz.dtype_like('arg')


@public
class BitXor(Filterable, Reduction):
    """Aggregate bitwise XOR operation.

    All elements in an integer column are XORed together. This can be used
    as a parity checksum of element values.

    Resources:

    * BigQuery [`BIT_XOR`](https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_xor)
    * MySQL [`BIT_XOR`](https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-xor)
    """  # noqa: E501

    arg = rlz.column(rlz.integer)
    output_dtype = rlz.dtype_like('arg')


@public
class Sum(Filterable, Reduction):
    arg = rlz.column(rlz.numeric)

    @immutable_property
    def output_dtype(self):
        if isinstance(self.arg, ir.BooleanValue):
            return dt.int64
        else:
            return self.arg.type().largest


@public
class Mean(Filterable, Reduction):
    arg = rlz.column(rlz.numeric)

    @immutable_property
    def output_dtype(self):
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg.type()
        else:
            return dt.float64

    def root_tables(self):
        return distinct_roots(self.arg)


@public
class Quantile(Reduction):
    arg = rlz.any
    quantile = rlz.strict_numeric
    interpolation = rlz.isin(
        {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
    )

    output_dtype = dt.float64


@public
class MultiQuantile(Quantile):
    arg = rlz.any
    quantile = rlz.value(dt.Array(dt.float64))
    interpolation = rlz.isin(
        {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
    )

    output_dtype = dt.Array(dt.float64)


@public
class VarianceBase(Filterable, Reduction):
    arg = rlz.column(rlz.numeric)
    how = rlz.isin({'sample', 'pop'})

    @immutable_property
    def output_dtype(self):
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg.type().largest
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

    left = rlz.column(rlz.numeric)
    right = rlz.column(rlz.numeric)
    how = rlz.isin({'sample', 'pop'})

    output_dtype = dt.float64


@public
class Covariance(Filterable, Reduction):
    """Covariance of a set of number pairs."""

    left = rlz.column(rlz.numeric)
    right = rlz.column(rlz.numeric)
    how = rlz.isin({'sample', 'pop'})

    output_dtype = dt.float64


@public
class Max(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like('arg')


@public
class Min(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like('arg')


@public
class ArgMax(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    key = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like("arg")


@public
class ArgMin(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    key = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like("arg")


@public
class ApproxCountDistinct(Filterable, Reduction):
    """Approximate number of unique values using HyperLogLog algorithm.

    Impala offers the NDV built-in function for this.
    """

    arg = rlz.column(rlz.any)

    # Impala 2.0 and higher returns a DOUBLE return ir.DoubleScalar
    output_dtype = dt.int64


@public
@deprecated(version='4.0', instead='use ApproxCountDistinct')
class HLLCardinality(ApproxCountDistinct):
    pass


@public
class GroupConcat(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    sep = rlz.string

    output_dtype = dt.string


@public
class ApproxMedian(Filterable, Reduction):
    """
    Compute the approximate median of a set of comparable values using the
    Count-Min-Sketch algorithm. Exposed in Impala using APPX_MEDIAN.
    """

    arg = rlz.column(rlz.any)
    output_dtype = rlz.dtype_like('arg')


@public
@deprecated(version="4.0", instead="use ApproxMedian")
class CMSMedian(ApproxMedian):
    pass


@public
class All(Reduction):
    arg = rlz.column(rlz.boolean)
    output_dtype = dt.boolean

    def negate(self):
        return NotAll(self.arg)


@public
class NotAll(All):
    def negate(self):
        return All(self.arg)


@public
class CountDistinct(Filterable, Reduction):
    arg = rlz.column(rlz.any)

    output_dtype = dt.int64


@public
class ArrayCollect(Reduction):
    arg = rlz.column(rlz.any)

    @immutable_property
    def output_dtype(self):
        return dt.Array(self.arg.type())


@public
class Any(Reduction, _Negatable):
    arg = rlz.column(rlz.boolean)

    output_dtype = dt.boolean

    def negate(self) -> NotAny:
        return NotAny(*self.args)


@public
class NotAny(Reduction, _Negatable):
    arg = rlz.column(rlz.boolean)

    output_dtype = dt.boolean

    def negate(self) -> Any:
        return Any(*self.args)
