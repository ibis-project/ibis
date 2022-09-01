from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Literal, Sequence

from public import public

from ibis.common.exceptions import IbisTypeError
from ibis.expr.types.core import _binop
from ibis.expr.types.generic import Column, Scalar, Value

if TYPE_CHECKING:
    from ibis.expr import types as ir


@public
class NumericValue(Value):
    @staticmethod
    def __negate_op__():
        from ibis.expr import operations as ops

        return ops.Negate

    def negate(self) -> NumericValue:
        """Negate a numeric expression.

        Returns
        -------
        NumericValue
            A numeric value expression
        """
        op = self.op()
        try:
            result = op.negate()
        except AttributeError:
            op_class = self.__negate_op__()
            result = op_class(self)

        return result.to_expr()

    def __neg__(self) -> NumericValue:
        """Negate `self`.

        Returns
        -------
        NumericValue
            `self` negated
        """
        return self.negate()

    def round(self, digits: int | IntegerValue | None = None) -> NumericValue:
        """Round values to an indicated number of decimal places.

        Parameters
        ----------
        digits
            The number of digits to round to.

            Here's how the `digits` parameter affects the expression output
            type:

            |   `digits`    | `self.type()` |  Output   |
            | :-----------: | :-----------: | :-------: |
            | `None` or `0` |   `decimal`   | `decimal` |
            |    Nonzero    |   `decimal`   | `decimal` |
            | `None` or `0` |   Floating    |  `int64`  |
            |    Nonzero    |   Floating    | `float64` |

        Returns
        -------
        NumericValue
            The rounded expression
        """
        from ibis.expr import operations as ops

        return ops.Round(self, digits).to_expr()

    def log(self, base: NumericValue | None = None) -> NumericValue:
        """Return the logarithm using a specified base.

        Parameters
        ----------
        base
            The base of the logarithm. If `None`, base `e` is used.

        Returns
        -------
        NumericValue
            Logarithm of `arg` with base `base`
        """
        from ibis.expr import operations as ops

        return ops.Log(self, base).to_expr()

    def clip(
        self,
        lower: NumericValue | None = None,
        upper: NumericValue | None = None,
    ) -> NumericValue:
        """Trim values outside of `lower` and `upper` bounds.

        Parameters
        ----------
        lower
            Lower bound
        upper
            Upper bound

        Returns
        -------
        NumericValue
            Clipped input
        """
        from ibis.expr import operations as ops

        if lower is None and upper is None:
            raise ValueError(
                "at least one of lower and upper must be provided"
            )

        return ops.Clip(self, lower, upper).to_expr()

    def abs(self) -> NumericValue:
        """Return the absolute value of `self`."""
        from ibis.expr import operations as ops

        return ops.Abs(self).to_expr()

    def ceil(self) -> DecimalValue | IntegerValue:
        """Return the ceiling of `self`."""
        from ibis.expr import operations as ops

        return ops.Ceil(self).to_expr()

    def degrees(self) -> NumericValue:
        """Compute the degrees of `self` radians."""
        from ibis.expr import operations as ops

        return ops.Degrees(self).to_expr()

    rad2deg = degrees

    def exp(self) -> NumericValue:
        r"""Compute $e^\texttt{self}$.

        Returns
        -------
        NumericValue
            $e^\texttt{self}$
        """
        from ibis.expr import operations as ops

        return ops.Exp(self).to_expr()

    def floor(self) -> DecimalValue | IntegerValue:
        """Return the floor of an expression."""
        from ibis.expr import operations as ops

        return ops.Floor(self).to_expr()

    def log2(self) -> NumericValue:
        r"""Compute $\log_{2}\left(\texttt{self}\right)$."""
        from ibis.expr import operations as ops

        return ops.Log2(self).to_expr()

    def log10(self) -> NumericValue:
        r"""Compute $\log_{10}\left(\texttt{self}\right)$."""
        from ibis.expr import operations as ops

        return ops.Log10(self).to_expr()

    def ln(self) -> NumericValue:
        r"""Compute $\ln\left(\texttt{self}\right)$."""
        from ibis.expr import operations as ops

        return ops.Ln(self).to_expr()

    def radians(self) -> NumericValue:
        """Compute radians from `self` degrees."""
        from ibis.expr import operations as ops

        return ops.Radians(self).to_expr()

    deg2rad = radians

    def sign(self) -> NumericValue:
        """Return the sign of the input."""
        from ibis.expr import operations as ops

        return ops.Sign(self).to_expr()

    def sqrt(self) -> NumericValue:
        """Compute the square root of `self`."""
        from ibis.expr import operations as ops

        return ops.Sqrt(self).to_expr()

    def nullifzero(self) -> NumericValue:
        """Return `NULL` if an expression is zero."""
        from ibis.expr import operations as ops

        return ops.NullIfZero(self).to_expr()

    def zeroifnull(self) -> NumericValue:
        """Return zero if an expression is `NULL`."""
        from ibis.expr import operations as ops

        return ops.ZeroIfNull(self).to_expr()

    def acos(self) -> NumericValue:
        """Compute the arc cosine of `self`."""
        from ibis.expr import operations as ops

        return ops.Acos(self).to_expr()

    def asin(self) -> NumericValue:
        """Compute the arc sine of `self`."""
        from ibis.expr import operations as ops

        return ops.Asin(self).to_expr()

    def atan(self) -> NumericValue:
        """Compute the arc tangent of `self`."""
        from ibis.expr import operations as ops

        return ops.Atan(self).to_expr()

    def atan2(self, other: NumericValue) -> NumericValue:
        """Compute the two-argument version of arc tangent."""
        from ibis.expr import operations as ops

        return ops.Atan2(self, other).to_expr()

    def cos(self) -> NumericValue:
        """Compute the cosine of `self`."""
        from ibis.expr import operations as ops

        return ops.Cos(self).to_expr()

    def cot(self) -> NumericValue:
        """Compute the cotangent of `self`."""
        from ibis.expr import operations as ops

        return ops.Cot(self).to_expr()

    def sin(self) -> NumericValue:
        """Compute the sine of `self`."""
        from ibis.expr import operations as ops

        return ops.Sin(self).to_expr()

    def tan(self) -> NumericValue:
        """Compute the tangent of `self`."""
        from ibis.expr import operations as ops

        return ops.Tan(self).to_expr()

    def __add__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Add `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.Add, self, other)

    add = radd = __radd__ = __add__

    def __sub__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Substract `other` from `self`."""
        from ibis.expr import operations as ops

        return _binop(ops.Subtract, self, other)

    sub = __sub__

    def __rsub__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Substract `self` from `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.Subtract, other, self)

    rsub = __rsub__

    def __mul__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Multiply `self` and `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.Multiply, self, other)

    mul = rmul = __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide `self` by `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.Divide, self, other)

    div = __div__ = __truediv__

    def __rtruediv__(
        self, other: NumericValue
    ) -> NumericValue | NotImplemented:
        """Divide `other` by `self`."""
        from ibis.expr import operations as ops

        return _binop(ops.Divide, other, self)

    rdiv = __rdiv__ = __rtruediv__

    def __floordiv__(
        self,
        other: NumericValue,
    ) -> NumericValue | NotImplemented:
        """Floor divide `self` by `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.FloorDivide, self, other)

    floordiv = __floordiv__

    def __rfloordiv__(
        self,
        other: NumericValue,
    ) -> NumericValue | NotImplemented:
        """Floor divide `other` by `self`."""
        from ibis.expr import operations as ops

        return _binop(ops.FloorDivide, other, self)

    rfloordiv = __rfloordiv__

    def __pow__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Raise `self` to the `other`th power."""
        from ibis.expr import operations as ops

        return _binop(ops.Power, self, other)

    pow = __pow__

    def __rpow__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Raise `other` to the `self`th power."""
        from ibis.expr import operations as ops

        return _binop(ops.Power, other, self)

    rpow = __rpow__

    def __mod__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Compute `self` modulo `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.Modulus, self, other)

    mod = __mod__

    def __rmod__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Compute `other` modulo `self`."""
        from ibis.expr import operations as ops

        return _binop(ops.Modulus, other, self)

    rmod = __rmod__

    def point(self, right: int | float | NumericValue) -> ir.PointValue:
        """Return a point constructed from the coordinate values.

        Constant coordinates result in construction of a `POINT` literal or
        column.

        Parameters
        ----------
        right
            Y coordinate

        Returns
        -------
        PointValue
            Points
        """
        from ibis.expr import operations as ops

        return ops.GeoPoint(self, right).to_expr()


@public
class NumericScalar(Scalar, NumericValue):
    pass  # noqa: E701,E302


@public
class NumericColumn(Column, NumericValue):
    def quantile(
        self,
        quantile: Sequence[NumericValue | float],
        interpolation: Literal[
            "linear",
            "lower",
            "higher",
            "midpoint",
            "nearest",
        ] = "linear",
    ) -> NumericScalar:
        """Return value at the given quantile.

        Parameters
        ----------
        quantile
            `0 <= quantile <= 1`, the quantile(s) to compute
        interpolation
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

            * linear: `i + (j - i) * fraction`, where `fraction` is the
              fractional part of the index surrounded by `i` and `j`.
            * lower: `i`.
            * higher: `j`.
            * nearest: `i` or `j` whichever is nearest.
            * midpoint: (`i` + `j`) / 2.

        Returns
        -------
        NumericScalar
            Quantile of the input
        """
        from ibis.expr import operations as ops

        if isinstance(quantile, collections.abc.Sequence):
            op = ops.MultiQuantile
        else:
            op = ops.Quantile
        return op(self, quantile, interpolation).to_expr()

    def std(
        self,
        where: ir.BooleanValue | None = None,
        how: Literal["sample", "pop"] = "sample",
    ) -> NumericScalar:
        """Return the standard deviation of a numeric column.

        Parameters
        ----------
        where
            Filter
        how
            Sample or population standard deviation

        Returns
        -------
        NumericScalar
            Standard deviation of `arg`
        """
        from ibis.expr import operations as ops

        return (
            ops.StandardDev(self, how=how, where=where).to_expr().name("std")
        )

    def var(
        self,
        where: ir.BooleanValue | None = None,
        how: Literal["sample", "pop"] = "sample",
    ) -> NumericScalar:
        """Return the variance of a numeric column.

        Parameters
        ----------
        where
            Filter
        how
            Sample or population variance

        Returns
        -------
        NumericScalar
            Standard deviation of `arg`
        """
        from ibis.expr import operations as ops

        return ops.Variance(self, how=how, where=where).to_expr().name("var")

    def corr(
        self,
        right: NumericColumn,
        where: ir.BooleanValue | None = None,
        how: Literal['sample', 'pop'] = 'sample',
    ) -> NumericScalar:
        """Return the correlation of two numeric columns.

        Parameters
        ----------
        right
            Numeric column
        where
            Filter
        how
            Population or sample correlation

        Returns
        -------
        NumericScalar
            The correlation of `left` and `right`
        """
        from ibis.expr import operations as ops

        return ops.Correlation(self, right, how=how, where=where).to_expr()

    def cov(
        self,
        right: NumericColumn,
        where: ir.BooleanValue | None = None,
        how: Literal['sample', 'pop'] = 'sample',
    ) -> NumericScalar:
        """Return the covariance of two numeric columns.

        Parameters
        ----------
        right
            Numeric column
        where
            Filter
        how
            Population or sample covariance

        Returns
        -------
        NumericScalar
            The covariance of `self` and `right`
        """
        from ibis.expr import operations as ops

        return ops.Covariance(self, right, how=how, where=where).to_expr()

    def mean(
        self,
        where: ir.BooleanValue | None = None,
    ) -> NumericScalar:
        """Return the mean of a numeric column.

        Parameters
        ----------
        where
            Filter

        Returns
        -------
        NumericScalar
            The mean of the input expression
        """
        from ibis.expr import operations as ops

        return ops.Mean(self, where=where).to_expr().name("mean")

    def cummean(self) -> NumericColumn:
        from ibis.expr import operations as ops

        return ops.CumulativeMean(self).to_expr()

    def sum(
        self,
        where: ir.BooleanValue | None = None,
    ) -> NumericScalar:
        """Return the sum of a numeric column.

        Parameters
        ----------
        where
            Filter

        Returns
        -------
        NumericScalar
            The sum of the input expression
        """
        from ibis.expr import operations as ops

        return ops.Sum(self, where=where).to_expr().name("sum")

    def cumsum(self) -> NumericColumn:
        from ibis.expr import operations as ops

        return ops.CumulativeSum(self).to_expr()

    def bucket(
        self,
        buckets: Sequence[int],
        closed: Literal["left", "right"] = "left",
        close_extreme: bool = True,
        include_under: bool = False,
        include_over: bool = False,
    ) -> ir.CategoryColumn:
        """Compute a discrete binning of a numeric array.

        Parameters
        ----------
        buckets
            List of buckets
        closed
            Which side of each interval is closed. For example:

            ```python
            buckets = [0, 100, 200]
            closed = "left"  # 100 falls in 2nd bucket
            closed = "right"  # 100 falls in 1st bucket
            ```
        close_extreme
            Whether the extreme values fall in the last bucket
        include_over
            Include values greater than the last bucket in the last bucket
        include_under
            Include values less than the first bucket in the first bucket

        Returns
        -------
        CategoryColumn
            A categorical column expression
        """
        from ibis.expr import operations as ops

        return ops.Bucket(
            self,
            buckets,
            closed=closed,
            close_extreme=close_extreme,
            include_under=include_under,
            include_over=include_over,
        ).to_expr()

    def histogram(
        self,
        nbins: int | None = None,
        binwidth: float | None = None,
        base: float | None = None,
        closed: Literal["left", "right"] = "left",
        aux_hash: str | None = None,
    ) -> ir.CategoryColumn:
        """Compute a histogram with fixed width bins.

        Parameters
        ----------
        nbins
            If supplied, will be used to compute the binwidth
        binwidth
            If not supplied, computed from the data (actual max and min values)
        base
            Histogram base
        closed
            Which side of each interval is closed
        aux_hash
            Auxiliary hash value to add to bucket names

        Returns
        -------
        CategoryColumn
            Coded value expression
        """
        from ibis.expr import operations as ops

        return ops.Histogram(
            self, nbins, binwidth, base, closed=closed, aux_hash=aux_hash
        ).to_expr()

    def summary(
        self,
        exact_nunique: bool = False,
        prefix: str = "",
        suffix: str = "",
    ) -> list[NumericScalar]:
        """Compute a set of summary metrics from the input numeric value
        expression.

        Parameters
        ----------
        exact_nunique
            Compute the exact number of distinct values. Typically slower if
            `True`.
        prefix
            String prefix for metric names
        suffix
            String suffix for metric names

        Returns
        -------
        list[NumericScalar]
            Metrics list
        """
        if exact_nunique:
            unique_metric = self.nunique().name("nunique")
        else:
            unique_metric = self.approx_nunique().name("approx_nunique")

        metrics = [
            self.count(),
            self.isnull().sum().name("nulls"),
            self.min(),
            self.max(),
            self.sum(),
            self.mean(),
            unique_metric,
        ]
        return [m.name(f"{prefix}{m.get_name()}{suffix}") for m in metrics]


@public
class IntegerValue(NumericValue):
    def to_timestamp(
        self,
        unit: Literal["s", "ms", "us"] = "s",
    ) -> ir.TimestampValue:
        """Convert an integral UNIX timestamp to a timestamp expression.

        Parameters
        ----------
        unit
            The resolution of `arg`

        Returns
        -------
        TimestampValue
            `self` converted to a timestamp
        """
        from ibis.expr import operations as ops

        return ops.TimestampFromUNIX(self, unit).to_expr()

    def to_interval(
        self,
        unit: Literal[
            "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"
        ] = "s",
    ) -> ir.IntervalValue:
        """Convert an integer to an interval.

        Parameters
        ----------
        unit
            Unit for the resulting interval

        Returns
        -------
        IntervalValue
            An interval in units of `unit`
        """
        from ibis.expr import operations as ops

        return ops.IntervalFromInteger(self, unit).to_expr()

    def convert_base(
        self,
        from_base: IntegerValue,
        to_base: IntegerValue,
    ) -> IntegerValue:
        """Convert an integer from one base to another.

        Parameters
        ----------
        from_base
            Numeric base of expression
        to_base
            New base

        Returns
        -------
        IntegerValue
            Converted expression
        """
        from ibis.expr import operations as ops

        return ops.BaseConvert(self, from_base, to_base).to_expr()

    def __and__(self, other: IntegerValue) -> IntegerValue | NotImplemented:
        """Bitwise and `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseAnd, self, other)

    __rand__ = __and__

    def __or__(self, other: IntegerValue) -> IntegerValue | NotImplemented:
        """Bitwise or `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseOr, self, other)

    __ror__ = __or__

    def __xor__(self, other: IntegerValue) -> IntegerValue | NotImplemented:
        """Bitwise xor `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseXor, self, other)

    __rxor__ = __xor__

    def __lshift__(self, other: IntegerValue) -> IntegerValue | NotImplemented:
        """Bitwise left shift `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseLeftShift, self, other)

    def __rlshift__(
        self, other: IntegerValue
    ) -> IntegerValue | NotImplemented:
        """Bitwise left shift `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseLeftShift, other, self)

    def __rshift__(self, other: IntegerValue) -> IntegerValue | NotImplemented:
        """Bitwise right shift `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseRightShift, self, other)

    def __rrshift__(
        self, other: IntegerValue
    ) -> IntegerValue | NotImplemented:
        """Bitwise right shift `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseRightShift, other, self)

    def __invert__(self) -> IntegerValue:
        """Bitwise not of `self`.

        Returns
        -------
        IntegerValue
            Inverted bits of `self`.
        """
        from ibis.expr import operations as ops

        try:
            node = ops.BitwiseNot(self)
        except (IbisTypeError, NotImplementedError):
            return NotImplemented
        else:
            return node.to_expr()


@public
class IntegerScalar(NumericScalar, IntegerValue):
    pass


@public
class IntegerColumn(NumericColumn, IntegerValue):
    def bit_and(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise and operator."""
        from ibis.expr import operations as ops

        return ops.BitAnd(self, where).to_expr().name("bit_and")

    def bit_or(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise or operator."""
        from ibis.expr import operations as ops

        return ops.BitOr(self, where).to_expr().name("bit_or")

    def bit_xor(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise exclusive or operator."""
        from ibis.expr import operations as ops

        return ops.BitXor(self, where).to_expr().name("bit_xor")


@public
class FloatingValue(NumericValue):
    def isnan(self) -> ir.BooleanValue:
        """Return whether the value is NaN."""
        from ibis.expr import operations as ops

        return ops.IsNan(self).to_expr()

    def isinf(self) -> ir.BooleanValue:
        """Return whether the value is infinity."""
        from ibis.expr import operations as ops

        return ops.IsInf(self).to_expr()


@public
class FloatingScalar(NumericScalar, FloatingValue):
    pass  # noqa: E701,E302


@public
class FloatingColumn(NumericColumn, FloatingValue):
    pass  # noqa: E701,E302


@public
class DecimalValue(NumericValue):
    def precision(self) -> IntegerValue:
        """Return the precision of `arg`.

        Returns
        -------
        IntegerValue
            The precision of the expression.
        """
        from ibis.expr import operations as ops

        return ops.DecimalPrecision(self).to_expr()

    def scale(self) -> IntegerValue:
        """Return the scale of `arg`.

        Returns
        -------
        IntegerValue
            The scale of the expression.
        """
        from ibis.expr import operations as ops

        return ops.DecimalScale(self).to_expr()


@public
class DecimalScalar(NumericScalar, DecimalValue):
    pass  # noqa: E701,E302


@public
class DecimalColumn(NumericColumn, DecimalValue):
    pass  # noqa: E701,E302
