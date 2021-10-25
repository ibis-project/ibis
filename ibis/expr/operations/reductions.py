import functools

from public import public

from .. import datatypes as dt
from .. import rules as rlz
from .. import types as ir
from ..signature import Argument as Arg
from .core import ValueOp


@public
class Reduction(ValueOp):
    _reduction = True


@public
class Count(Reduction):
    arg = Arg(rlz.one_of((rlz.column(rlz.any), rlz.table)))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return functools.partial(ir.IntegerScalar, dtype=dt.int64)


@public
class Arbitrary(Reduction):
    arg = Arg(rlz.column(rlz.any))
    how = Arg(rlz.isin({'first', 'last', 'heavy'}), default=None)
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


@public
class BitAnd(Reduction):
    """Aggregate bitwise AND operation.

    All elements in an integer column are ANDed together. This can be used
    to determine which bit flags are set on all elements.

    Resources:

    * `BigQuery BIT_AND
      <https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_and>`_
    * `MySQL BIT_AND
      <https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-and>`_
    """

    arg = Arg(rlz.column(rlz.integer))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


@public
class BitOr(Reduction):
    """Aggregate bitwise OR operation.

    All elements in an integer column are ORed together. This can be used
    to determine which bit flags are set on any element.

    Resources:

    * `BigQuery BIT_OR
      <https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_or>`_
    * `MySQL BIT_OR
      <https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-or>`_
    """

    arg = Arg(rlz.column(rlz.integer))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


@public
class BitXor(Reduction):
    """Aggregate bitwise XOR operation.

    All elements in an integer column are XORed together. This can be used
    as a parity checksum of element values.

    Resources:

    * `BigQuery BIT_XOR
      <https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#bit_xor>`_
    * `MySQL BIT_XOR
      <https://dev.mysql.com/doc/refman/5.7/en/aggregate-functions.html#function_bit-xor>`_
    """

    arg = Arg(rlz.column(rlz.integer))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


@public
class Sum(Reduction):
    arg = Arg(rlz.column(rlz.numeric))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest
        return dtype.scalar_type()


@public
class Mean(Reduction):
    arg = Arg(rlz.column(rlz.numeric))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type()
        else:
            dtype = dt.float64
        return dtype.scalar_type()


@public
class Quantile(Reduction):
    arg = Arg(rlz.any)
    quantile = Arg(rlz.strict_numeric)
    interpolation = Arg(
        rlz.isin({'linear', 'lower', 'higher', 'midpoint', 'nearest'}),
        default='linear',
    )

    def output_type(self):
        return dt.float64.scalar_type()


@public
class MultiQuantile(Quantile):
    arg = Arg(rlz.any)
    quantile = Arg(rlz.value(dt.Array(dt.float64)))
    interpolation = Arg(
        rlz.isin({'linear', 'lower', 'higher', 'midpoint', 'nearest'}),
        default='linear',
    )

    def output_type(self):
        return dt.Array(dt.float64).scalar_type()


@public
class VarianceBase(Reduction):
    arg = Arg(rlz.column(rlz.numeric))
    how = Arg(rlz.isin({'sample', 'pop'}), default=None)
    where = Arg(rlz.boolean, default=None)

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
class Correlation(Reduction):
    """Coefficient of correlation of a set of number pairs."""

    left = Arg(rlz.column(rlz.numeric))
    right = Arg(rlz.column(rlz.numeric))
    how = Arg(rlz.isin({'sample', 'pop'}), default=None)
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return dt.float64.scalar_type()


@public
class Covariance(Reduction):
    """Covariance of a set of number pairs."""

    left = Arg(rlz.column(rlz.numeric))
    right = Arg(rlz.column(rlz.numeric))
    how = Arg(rlz.isin({'sample', 'pop'}), default=None)
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return dt.float64.scalar_type()


@public
class Max(Reduction):
    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


@public
class Min(Reduction):
    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


@public
class HLLCardinality(Reduction):
    """Approximate number of unique values using HyperLogLog algorithm.

    Impala offers the NDV built-in function for this.
    """

    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        # return ir.DoubleScalar
        return functools.partial(ir.IntegerScalar, dtype=dt.int64)


@public
class GroupConcat(Reduction):
    arg = Arg(rlz.column(rlz.any))
    sep = Arg(rlz.string, default=',')
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return dt.string.scalar_type()


@public
class CMSMedian(Reduction):
    """
    Compute the approximate median of a set of comparable values using the
    Count-Min-Sketch algorithm. Exposed in Impala using APPX_MEDIAN.
    """

    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


@public
class All(ValueOp):
    arg = Arg(rlz.column(rlz.boolean))
    output_type = rlz.scalar_like('arg')
    _reduction = True

    def negate(self):
        return NotAll(self.arg)


@public
class NotAll(All):
    def negate(self):
        return All(self.arg)


@public
class CountDistinct(Reduction):
    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return dt.int64.scalar_type()


@public
class ArrayCollect(Reduction):
    arg = Arg(rlz.column(rlz.any))

    def output_type(self):
        dtype = dt.Array(self.arg.type())
        return dtype.scalar_type()
