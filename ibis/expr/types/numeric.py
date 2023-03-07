from __future__ import annotations

import collections
import functools
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

from public import public

import ibis.expr.operations as ops
from ibis import util
from ibis.common.exceptions import IbisTypeError
from ibis.expr.types.core import _binop
from ibis.expr.types.generic import Column, Scalar, Value

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class NumericValue(Value):
    @staticmethod
    def __negate_op__():
        # TODO(kszucs): do we need this?
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
        if lower is None and upper is None:
            raise ValueError("at least one of lower and upper must be provided")

        return ops.Clip(self, lower, upper).to_expr()

    def abs(self) -> NumericValue:
        """Return the absolute value of `self`."""
        return ops.Abs(self).to_expr()

    def ceil(self) -> DecimalValue | IntegerValue:
        """Return the ceiling of `self`."""
        return ops.Ceil(self).to_expr()

    def degrees(self) -> NumericValue:
        """Compute the degrees of `self` radians."""
        return ops.Degrees(self).to_expr()

    rad2deg = degrees

    def exp(self) -> NumericValue:
        r"""Compute $e^\texttt{self}$.

        Returns
        -------
        NumericValue
            $e^\texttt{self}$
        """
        return ops.Exp(self).to_expr()

    def floor(self) -> DecimalValue | IntegerValue:
        """Return the floor of an expression."""
        return ops.Floor(self).to_expr()

    def log2(self) -> NumericValue:
        r"""Compute $\log_{2}\left(\texttt{self}\right)$."""
        return ops.Log2(self).to_expr()

    def log10(self) -> NumericValue:
        r"""Compute $\log_{10}\left(\texttt{self}\right)$."""
        return ops.Log10(self).to_expr()

    def ln(self) -> NumericValue:
        r"""Compute $\ln\left(\texttt{self}\right)$."""
        return ops.Ln(self).to_expr()

    def radians(self) -> NumericValue:
        """Compute radians from `self` degrees."""
        return ops.Radians(self).to_expr()

    deg2rad = radians

    def sign(self) -> NumericValue:
        """Return the sign of the input."""
        return ops.Sign(self).to_expr()

    def sqrt(self) -> NumericValue:
        """Compute the square root of `self`."""
        return ops.Sqrt(self).to_expr()

    def nullifzero(self) -> NumericValue:
        """Return `NULL` if an expression is zero."""
        return ops.NullIfZero(self).to_expr()

    def zeroifnull(self) -> NumericValue:
        """Return zero if an expression is `NULL`."""
        return ops.ZeroIfNull(self).to_expr()

    def acos(self) -> NumericValue:
        """Compute the arc cosine of `self`."""
        return ops.Acos(self).to_expr()

    def asin(self) -> NumericValue:
        """Compute the arc sine of `self`."""
        return ops.Asin(self).to_expr()

    def atan(self) -> NumericValue:
        """Compute the arc tangent of `self`."""
        return ops.Atan(self).to_expr()

    def atan2(self, other: NumericValue) -> NumericValue:
        """Compute the two-argument version of arc tangent."""
        return ops.Atan2(self, other).to_expr()

    def cos(self) -> NumericValue:
        """Compute the cosine of `self`."""
        return ops.Cos(self).to_expr()

    def cot(self) -> NumericValue:
        """Compute the cotangent of `self`."""
        return ops.Cot(self).to_expr()

    def sin(self) -> NumericValue:
        """Compute the sine of `self`."""
        return ops.Sin(self).to_expr()

    def tan(self) -> NumericValue:
        """Compute the tangent of `self`."""
        return ops.Tan(self).to_expr()

    def __add__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Add `self` with `other`."""
        return _binop(ops.Add, self, other)

    add = radd = __radd__ = __add__

    def __sub__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Substract `other` from `self`."""
        return _binop(ops.Subtract, self, other)

    sub = __sub__

    def __rsub__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Substract `self` from `other`."""
        return _binop(ops.Subtract, other, self)

    rsub = __rsub__

    def __mul__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Multiply `self` and `other`."""
        return _binop(ops.Multiply, self, other)

    mul = rmul = __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide `self` by `other`."""
        return _binop(ops.Divide, self, other)

    div = __div__ = __truediv__

    def __rtruediv__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Divide `other` by `self`."""
        return _binop(ops.Divide, other, self)

    rdiv = __rdiv__ = __rtruediv__

    def __floordiv__(
        self,
        other: NumericValue,
    ) -> NumericValue | NotImplemented:
        """Floor divide `self` by `other`."""
        return _binop(ops.FloorDivide, self, other)

    floordiv = __floordiv__

    def __rfloordiv__(
        self,
        other: NumericValue,
    ) -> NumericValue | NotImplemented:
        """Floor divide `other` by `self`."""
        return _binop(ops.FloorDivide, other, self)

    rfloordiv = __rfloordiv__

    def __pow__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Raise `self` to the `other`th power."""
        return _binop(ops.Power, self, other)

    pow = __pow__

    def __rpow__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Raise `other` to the `self`th power."""
        return _binop(ops.Power, other, self)

    rpow = __rpow__

    def __mod__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Compute `self` modulo `other`."""
        return _binop(ops.Modulus, self, other)

    mod = __mod__

    def __rmod__(self, other: NumericValue) -> NumericValue | NotImplemented:
        """Compute `other` modulo `self`."""

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
        return ops.GeoPoint(self, right).to_expr()


@public
class NumericScalar(Scalar, NumericValue):
    pass


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
        ]
        | None = None,
        where: ir.BooleanValue | None = None,
    ) -> NumericScalar:
        """Return value at the given quantile.

        Parameters
        ----------
        quantile
            `0 <= quantile <= 1`, the quantile(s) to compute
        interpolation
            !!! warning "This parameter is backend dependent and may have no effect"

            This parameter specifies the interpolation method to use, when the
            desired quantile lies between two data points `i` and `j`:

            * linear: `i + (j - i) * fraction`, where `fraction` is the
              fractional part of the index surrounded by `i` and `j`.
            * lower: `i`.
            * higher: `j`.
            * nearest: `i` or `j` whichever is nearest.
            * midpoint: (`i` + `j`) / 2.
        where
            Boolean filter for input values

        Returns
        -------
        NumericScalar
            Quantile of the input
        """
        if isinstance(quantile, collections.abc.Sequence):
            op = ops.MultiQuantile
        else:
            op = ops.Quantile
        return op(self, quantile, interpolation, where=where).to_expr()

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
        return ops.StandardDev(self, how=how, where=where).to_expr()

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
        return ops.Variance(self, how=how, where=where).to_expr()

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
        # TODO(kszucs): remove the alias from the reduction method in favor
        # of default name generated by ops.Value operations
        return ops.Mean(self, where=where).to_expr()

    def cummean(self) -> NumericColumn:
        """Return the cumulative mean of the input."""
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
        return ops.Sum(self, where=where).to_expr()

    def cumsum(self) -> NumericColumn:
        """Return the cumulative sum of the input."""
        return ops.CumulativeSum(self).to_expr()

    def bucket(
        self,
        buckets: Sequence[int],
        closed: Literal["left", "right"] = "left",
        close_extreme: bool = True,
        include_under: bool = False,
        include_over: bool = False,
    ) -> ir.IntegerColumn:
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
        IntegerColumn
            A categorical column expression
        """
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
        eps: float = 1e-13,
    ):
        """Compute a histogram with fixed width bins.

        Parameters
        ----------
        nbins
            If supplied, will be used to compute the binwidth
        binwidth
            If not supplied, computed from the data (actual max and min values)
        base
            The value of the first histogram bin. Defaults to the minimum value
            of `column`.
        eps
            Allowed floating point epsilon for histogram base

        Returns
        -------
        Column
            Bucketed column
        """

        if nbins is not None and binwidth is not None:
            raise ValueError(
                f"Cannot pass both `nbins` (got {nbins}) and `binwidth` (got {binwidth})"
            )

        if binwidth is None or base is None:
            import ibis

            if nbins is None:
                raise ValueError("`nbins` is required if `binwidth` is not provided")

            empty_window = ibis.window()

            if base is None:
                base = self.min().over(empty_window) - eps

            binwidth = (self.max().over(empty_window) - base) / (nbins - 1)

        return ((self - base) / binwidth).floor()

    @util.deprecated(
        instead="Reach out at https://github.com/ibis-project/ibis if you'd like this API to remain.",
        as_of="5.0",
        removed_in="6.0",
    )
    def summary(
        self,
        exact_nunique: bool = False,
        prefix: str = "",
        suffix: str = "",
    ) -> list[NumericScalar]:
        """Compute summary metrics from the input numeric expression.

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
            self.count().name("count"),
            self.isnull().sum().name("nulls"),
            self.min().name("min"),
            self.max().name("max"),
            self.sum().name("sum"),
            self.mean().name("mean"),
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
        return ops.TimestampFromUNIX(self, unit).to_expr()

    def to_interval(
        self,
        unit: Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"] = "s",
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

    def __rlshift__(self, other: IntegerValue) -> IntegerValue | NotImplemented:
        """Bitwise left shift `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseLeftShift, other, self)

    def __rshift__(self, other: IntegerValue) -> IntegerValue | NotImplemented:
        """Bitwise right shift `self` with `other`."""
        from ibis.expr import operations as ops

        return _binop(ops.BitwiseRightShift, self, other)

    def __rrshift__(self, other: IntegerValue) -> IntegerValue | NotImplemented:
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

    def label(self, labels: Iterable[str], nulls: str | None = None) -> ir.StringValue:
        """Label a set of integer values with strings.

        Parameters
        ----------
        labels
            An iterable of string labels. Each integer value in `self` will be mapped to
            a value in `labels`.
        nulls
            String label to use for `NULL` values

        Returns
        -------
        StringValue
            `self` labeled with `labels`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [0, 1, 0, 2]})
        >>> t.select(t.a, labeled=t.a.label(["a", "b", "c"]))
        ┏━━━━━━━┳━━━━━━━━━┓
        ┃ a     ┃ labeled ┃
        ┡━━━━━━━╇━━━━━━━━━┩
        │ int64 │ string  │
        ├───────┼─────────┤
        │     0 │ a       │
        │     1 │ b       │
        │     0 │ a       │
        │     2 │ c       │
        └───────┴─────────┘
        """
        return (
            functools.reduce(
                lambda stmt, inputs: stmt.when(*inputs), enumerate(labels), self.case()
            )
            .else_(nulls)
            .end()
        )


@public
class IntegerScalar(NumericScalar, IntegerValue):
    pass


@public
class IntegerColumn(NumericColumn, IntegerValue):
    def bit_and(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise and operator."""
        return ops.BitAnd(self, where).to_expr()

    def bit_or(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise or operator."""
        return ops.BitOr(self, where).to_expr()

    def bit_xor(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise exclusive or operator."""
        return ops.BitXor(self, where).to_expr()


@public
class FloatingValue(NumericValue):
    def isnan(self) -> ir.BooleanValue:
        """Return whether the value is NaN."""
        return ops.IsNan(self).to_expr()

    def isinf(self) -> ir.BooleanValue:
        """Return whether the value is infinity."""
        return ops.IsInf(self).to_expr()


@public
class FloatingScalar(NumericScalar, FloatingValue):
    pass


@public
class FloatingColumn(NumericColumn, FloatingValue):
    pass


@public
class DecimalValue(NumericValue):
    pass


@public
class DecimalScalar(NumericScalar, DecimalValue):
    pass


@public
class DecimalColumn(NumericColumn, DecimalValue):
    pass
