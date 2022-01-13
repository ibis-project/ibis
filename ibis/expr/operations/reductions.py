import functools

from public import public

from .. import datatypes as dt
from .. import rules as rlz
from .. import types as ir
from .core import ValueOp


@public
class Reduction(ValueOp):
    _reduction = True


class Filterable(ValueOp):
    where = rlz.optional(rlz.boolean)


@public
class Count(Filterable, Reduction):
    arg = rlz.one_of((rlz.column(rlz.any), rlz.table))

    def output_type(self):
        return functools.partial(ir.IntegerScalar, dtype=dt.int64)


@public
class Arbitrary(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    how = rlz.optional(rlz.isin({'first', 'last', 'heavy'}))
    output_type = rlz.scalar_like('arg')


@public
class BitAnd(Filterable, Reduction):
    """Aggregate bitwise AND operation.

    All elements in an integer column are ANDed together. This can be used
    to determine which bit flags are set on all elements.

    Resources:

    * `BigQuery BIT_AND
      <https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_and>`_
    * `MySQL BIT_AND
      <https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-and>`_
    """

    arg = rlz.column(rlz.integer)
    output_type = rlz.scalar_like('arg')


@public
class BitOr(Filterable, Reduction):
    """Aggregate bitwise OR operation.

    All elements in an integer column are ORed together. This can be used
    to determine which bit flags are set on any element.

    Resources:

    * `BigQuery BIT_OR
      <https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_or>`_
    * `MySQL BIT_OR
      <https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-or>`_
    """

    arg = rlz.column(rlz.integer)
    output_type = rlz.scalar_like('arg')


@public
class BitXor(Filterable, Reduction):
    """Aggregate bitwise XOR operation.

    All elements in an integer column are XORed together. This can be used
    as a parity checksum of element values.

    Resources:

    * `BigQuery BIT_XOR
      <https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_xor>`_
    * `MySQL BIT_XOR
      <https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-xor>`_
    """

    arg = rlz.column(rlz.integer)
    output_type = rlz.scalar_like('arg')


@public
class Sum(Filterable, Reduction):
    arg = rlz.column(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest
        return dtype.scalar_type()


@public
class Mean(Filterable, Reduction):
    arg = rlz.column(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type()
        else:
            dtype = dt.float64
        return dtype.scalar_type()


@public
class Quantile(Reduction):
    arg = rlz.any
    quantile = rlz.strict_numeric
    interpolation = rlz.isin(
        {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
    )

    def output_type(self):
        return dt.float64.scalar_type()


@public
class MultiQuantile(Quantile):
    arg = rlz.any
    quantile = rlz.value(dt.Array(dt.float64))
    interpolation = rlz.isin(
        {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
    )

    def output_type(self):
        return dt.Array(dt.float64).scalar_type()


@public
class VarianceBase(Filterable, Reduction):
    arg = rlz.column(rlz.numeric)
    how = rlz.isin({'sample', 'pop'})

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest
        else:
            dtype = dt.float64
        return dtype.scalar_type()


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

    def output_type(self):
        return dt.float64.scalar_type()


@public
class Covariance(Filterable, Reduction):
    """Covariance of a set of number pairs."""

    left = rlz.column(rlz.numeric)
    right = rlz.column(rlz.numeric)
    how = rlz.isin({'sample', 'pop'})

    def output_type(self):
        return dt.float64.scalar_type()


@public
class Max(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    output_type = rlz.scalar_like('arg')


@public
class Min(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    output_type = rlz.scalar_like('arg')


@public
class HLLCardinality(Filterable, Reduction):
    """Approximate number of unique values using HyperLogLog algorithm.

    Impala offers the NDV built-in function for this.
    """

    arg = rlz.column(rlz.any)

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        # return ir.DoubleScalar
        return functools.partial(ir.IntegerScalar, dtype=dt.int64)


@public
class GroupConcat(Filterable, Reduction):
    arg = rlz.column(rlz.any)
    sep = rlz.string

    def output_type(self):
        return dt.string.scalar_type()


@public
class CMSMedian(Filterable, Reduction):
    """
    Compute the approximate median of a set of comparable values using the
    Count-Min-Sketch algorithm. Exposed in Impala using APPX_MEDIAN.
    """

    arg = rlz.column(rlz.any)
    output_type = rlz.scalar_like('arg')


@public
class All(Reduction):
    arg = rlz.column(rlz.boolean)
    output_type = rlz.scalar_like('arg')

    def negate(self):
        return NotAll(self.arg)


@public
class NotAll(All):
    def negate(self):
        return All(self.arg)


@public
class CountDistinct(Filterable, Reduction):
    arg = rlz.column(rlz.any)

    def output_type(self):
        return dt.int64.scalar_type()


@public
class ArrayCollect(Reduction):
    arg = rlz.column(rlz.any)

    def output_type(self):
        dtype = dt.Array(self.arg.type())
        return dtype.scalar_type()
