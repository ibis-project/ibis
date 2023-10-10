from __future__ import annotations

import collections
import functools
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

from public import public

import ibis
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.negate()
        ┏━━━━━━━━━━━━━━━━┓
        ┃ Negate(values) ┃
        ┡━━━━━━━━━━━━━━━━┩
        │ int64          │
        ├────────────────┤
        │              1 │
        │              0 │
        │             -1 │
        └────────────────┘
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

            - `digits` is `False`-y; `self.type()` is `decimal` → `decimal`
            -   `digits` is nonzero; `self.type()` is `decimal` → `decimal`
            - `digits` is `False`-y; `self.type()` is Floating  → `int64`
            -   `digits` is nonzero; `self.type()` is Floating  → `float64`

        Returns
        -------
        NumericValue
            The rounded expression

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [1.22, 1.64, 2.15, 2.54]})
        >>> t
        ┏━━━━━━━━━┓
        ┃ values  ┃
        ┡━━━━━━━━━┩
        │ float64 │
        ├─────────┤
        │    1.22 │
        │    1.64 │
        │    2.15 │
        │    2.54 │
        └─────────┘
        >>> t.values.round()
        ┏━━━━━━━━━━━━━━━┓
        ┃ Round(values) ┃
        ┡━━━━━━━━━━━━━━━┩
        │ int64         │
        ├───────────────┤
        │             1 │
        │             2 │
        │             2 │
        │             3 │
        └───────────────┘
        >>> t.values.round(digits=1)
        ┏━━━━━━━━━━━━━━━━━━┓
        ┃ Round(values, 1) ┃
        ┡━━━━━━━━━━━━━━━━━━┩
        │ float64          │
        ├──────────────────┤
        │              1.2 │
        │              1.6 │
        │              2.2 │
        │              2.5 │
        └──────────────────┘
        """
        return ops.Round(self, digits).to_expr()

    def log(self, base: NumericValue | None = None) -> NumericValue:
        r"""Compute $\log_{\texttt{base}}\left(\texttt{self}\right)$.

        Parameters
        ----------
        base
            The base of the logarithm. If `None`, base `e` is used.

        Returns
        -------
        NumericValue
            Logarithm of `arg` with base `base`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> from math import e
        >>> t = ibis.memtable({"values": [e, e**2, e**3]})
        >>> t.values.log()
        ┏━━━━━━━━━━━━━┓
        ┃ Log(values) ┃
        ┡━━━━━━━━━━━━━┩
        │ float64     │
        ├─────────────┤
        │         1.0 │
        │         2.0 │
        │         3.0 │
        └─────────────┘


        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [10, 100, 1000]})
        >>> t.values.log(base=10)
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ Log(values, 10) ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ float64         │
        ├─────────────────┤
        │             1.0 │
        │             2.0 │
        │             3.0 │
        └─────────────────┘
        """
        return ops.Log(self, base).to_expr()

    def clip(
        self,
        lower: NumericValue | None = None,
        upper: NumericValue | None = None,
    ) -> NumericValue:
        """Trim values outside of `lower` and `upper` bounds.

        `NULL` values are preserved and are not replaced with bounds.

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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {"values": [None, 2, 3, None, 5, None, None, 8]},
        ...     schema=dict(values="int"),
        ... )
        >>> t.values.clip(lower=3, upper=6)
        ┏━━━━━━━━━━━━━━━━━━━━┓
        ┃ Clip(values, 3, 6) ┃
        ┡━━━━━━━━━━━━━━━━━━━━┩
        │ int64              │
        ├────────────────────┤
        │               NULL │
        │                  3 │
        │                  3 │
        │               NULL │
        │                  5 │
        │               NULL │
        │               NULL │
        │                  6 │
        └────────────────────┘
        """
        if lower is None and upper is None:
            raise ValueError("at least one of lower and upper must be provided")

        return ops.Clip(self, lower, upper).to_expr()

    def abs(self) -> NumericValue:
        """Return the absolute value of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 2, -3, 4]})
        >>> t.values.abs()
        ┏━━━━━━━━━━━━━┓
        ┃ Abs(values) ┃
        ┡━━━━━━━━━━━━━┩
        │ int64       │
        ├─────────────┤
        │           1 │
        │           2 │
        │           3 │
        │           4 │
        └─────────────┘
        """
        return ops.Abs(self).to_expr()

    def ceil(self) -> DecimalValue | IntegerValue:
        """Return the ceiling of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [1, 1.1, 2, 2.1, 3.3]})
        >>> t.values.ceil()
        ┏━━━━━━━━━━━━━━┓
        ┃ Ceil(values) ┃
        ┡━━━━━━━━━━━━━━┩
        │ int64        │
        ├──────────────┤
        │            1 │
        │            2 │
        │            2 │
        │            3 │
        │            4 │
        └──────────────┘
        """
        return ops.Ceil(self).to_expr()

    def degrees(self) -> NumericValue:
        """Compute the degrees of `self` radians.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> from math import pi
        >>> t = ibis.memtable({"values": [0, pi / 2, pi, 3 * pi / 2, 2 * pi]})
        >>> t.values.degrees()
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ Degrees(values) ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ float64         │
        ├─────────────────┤
        │             0.0 │
        │            90.0 │
        │           180.0 │
        │           270.0 │
        │           360.0 │
        └─────────────────┘
        """
        return ops.Degrees(self).to_expr()

    rad2deg = degrees

    def exp(self) -> NumericValue:
        r"""Compute $e^\texttt{self}$.

        Returns
        -------
        NumericValue
            $e^\texttt{self}$

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": range(4)})
        >>> t.values.exp()
        ┏━━━━━━━━━━━━━┓
        ┃ Exp(values) ┃
        ┡━━━━━━━━━━━━━┩
        │ float64     │
        ├─────────────┤
        │    1.000000 │
        │    2.718282 │
        │    7.389056 │
        │   20.085537 │
        └─────────────┘
        """
        return ops.Exp(self).to_expr()

    def floor(self) -> DecimalValue | IntegerValue:
        """Return the floor of an expression.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [1, 1.1, 2, 2.1, 3.3]})
        >>> t.values.floor()
        ┏━━━━━━━━━━━━━━━┓
        ┃ Floor(values) ┃
        ┡━━━━━━━━━━━━━━━┩
        │ int64         │
        ├───────────────┤
        │             1 │
        │             1 │
        │             2 │
        │             2 │
        │             3 │
        └───────────────┘

        """
        return ops.Floor(self).to_expr()

    def log2(self) -> NumericValue:
        r"""Compute $\log_{2}\left(\texttt{self}\right)$.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [1, 2, 4, 8]})
        >>> t.values.log2()
        ┏━━━━━━━━━━━━━━┓
        ┃ Log2(values) ┃
        ┡━━━━━━━━━━━━━━┩
        │ float64      │
        ├──────────────┤
        │          0.0 │
        │          1.0 │
        │          2.0 │
        │          3.0 │
        └──────────────┘
        """
        return ops.Log2(self).to_expr()

    def log10(self) -> NumericValue:
        r"""Compute $\log_{10}\left(\texttt{self}\right)$.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [1, 10, 100]})
        >>> t.values.log10()
        ┏━━━━━━━━━━━━━━━┓
        ┃ Log10(values) ┃
        ┡━━━━━━━━━━━━━━━┩
        │ float64       │
        ├───────────────┤
        │           0.0 │
        │           1.0 │
        │           2.0 │
        └───────────────┘
        """
        return ops.Log10(self).to_expr()

    def ln(self) -> NumericValue:
        r"""Compute $\ln\left(\texttt{self}\right)$.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [1, 2.718281828, 3]})
        >>> t.values.ln()
        ┏━━━━━━━━━━━━┓
        ┃ Ln(values) ┃
        ┡━━━━━━━━━━━━┩
        │ float64    │
        ├────────────┤
        │   0.000000 │
        │   1.000000 │
        │   1.098612 │
        └────────────┘
        """
        return ops.Ln(self).to_expr()

    def radians(self) -> NumericValue:
        """Compute radians from `self` degrees.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [0, 90, 180, 270, 360]})
        >>> t.values.radians()
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ Radians(values) ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ float64         │
        ├─────────────────┤
        │        0.000000 │
        │        1.570796 │
        │        3.141593 │
        │        4.712389 │
        │        6.283185 │
        └─────────────────┘
        """
        return ops.Radians(self).to_expr()

    deg2rad = radians

    def sign(self) -> NumericValue:
        """Return the sign of the input.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 2, -3, 4]})
        >>> t.values.sign()
        ┏━━━━━━━━━━━━━━┓
        ┃ Sign(values) ┃
        ┡━━━━━━━━━━━━━━┩
        │ int64        │
        ├──────────────┤
        │           -1 │
        │            1 │
        │           -1 │
        │            1 │
        └──────────────┘
        """
        return ops.Sign(self).to_expr()

    def sqrt(self) -> NumericValue:
        """Compute the square root of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [1, 4, 9, 16]})
        >>> t.values.sqrt()
        ┏━━━━━━━━━━━━━━┓
        ┃ Sqrt(values) ┃
        ┡━━━━━━━━━━━━━━┩
        │ float64      │
        ├──────────────┤
        │          1.0 │
        │          2.0 │
        │          3.0 │
        │          4.0 │
        └──────────────┘
        """
        return ops.Sqrt(self).to_expr()

    @util.deprecated(instead="use nullif(0)", as_of="7.0", removed_in="8.0")
    def nullifzero(self) -> NumericValue:
        """DEPRECATED: Use `nullif(0)` instead."""
        return self.nullif(0)

    @util.deprecated(instead="use fillna(0)", as_of="7.0", removed_in="8.0")
    def zeroifnull(self) -> NumericValue:
        """DEPRECATED: Use `fillna(0)` instead."""
        return self.fillna(0)

    def acos(self) -> NumericValue:
        """Compute the arc cosine of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.acos()
        ┏━━━━━━━━━━━━━━┓
        ┃ Acos(values) ┃
        ┡━━━━━━━━━━━━━━┩
        │ float64      │
        ├──────────────┤
        │     3.141593 │
        │     1.570796 │
        │     0.000000 │
        └──────────────┘

        """
        return ops.Acos(self).to_expr()

    def asin(self) -> NumericValue:
        """Compute the arc sine of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.asin()
        ┏━━━━━━━━━━━━━━┓
        ┃ Asin(values) ┃
        ┡━━━━━━━━━━━━━━┩
        │ float64      │
        ├──────────────┤
        │    -1.570796 │
        │     0.000000 │
        │     1.570796 │
        └──────────────┘
        """
        return ops.Asin(self).to_expr()

    def atan(self) -> NumericValue:
        """Compute the arc tangent of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.atan()
        ┏━━━━━━━━━━━━━━┓
        ┃ Atan(values) ┃
        ┡━━━━━━━━━━━━━━┩
        │ float64      │
        ├──────────────┤
        │    -0.785398 │
        │     0.000000 │
        │     0.785398 │
        └──────────────┘
        """
        return ops.Atan(self).to_expr()

    def atan2(self, other: NumericValue) -> NumericValue:
        """Compute the two-argument version of arc tangent.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.atan2(0)
        ┏━━━━━━━━━━━━━━━━━━┓
        ┃ Atan2(values, 0) ┃
        ┡━━━━━━━━━━━━━━━━━━┩
        │ float64          │
        ├──────────────────┤
        │        -1.570796 │
        │         0.000000 │
        │         1.570796 │
        └──────────────────┘
        """
        return ops.Atan2(self, other).to_expr()

    def cos(self) -> NumericValue:
        """Compute the cosine of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.cos()
        ┏━━━━━━━━━━━━━┓
        ┃ Cos(values) ┃
        ┡━━━━━━━━━━━━━┩
        │ float64     │
        ├─────────────┤
        │    0.540302 │
        │    1.000000 │
        │    0.540302 │
        └─────────────┘
        """
        return ops.Cos(self).to_expr()

    def cot(self) -> NumericValue:
        """Compute the cotangent of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.cot()
        ┏━━━━━━━━━━━━━┓
        ┃ Cot(values) ┃
        ┡━━━━━━━━━━━━━┩
        │ float64     │
        ├─────────────┤
        │   -0.642093 │
        │         inf │
        │    0.642093 │
        └─────────────┘
        """
        return ops.Cot(self).to_expr()

    def sin(self) -> NumericValue:
        """Compute the sine of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.sin()
        ┏━━━━━━━━━━━━━┓
        ┃ Sin(values) ┃
        ┡━━━━━━━━━━━━━┩
        │ float64     │
        ├─────────────┤
        │   -0.841471 │
        │    0.000000 │
        │    0.841471 │
        └─────────────┘
        """
        return ops.Sin(self).to_expr()

    def tan(self) -> NumericValue:
        """Compute the tangent of `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"values": [-1, 0, 1]})
        >>> t.values.tan()
        ┏━━━━━━━━━━━━━┓
        ┃ Tan(values) ┃
        ┡━━━━━━━━━━━━━┩
        │ float64     │
        ├─────────────┤
        │   -1.557408 │
        │    0.000000 │
        │    1.557408 │
        └─────────────┘
        """
        return ops.Tan(self).to_expr()

    def __add__(self, other: NumericValue) -> NumericValue:
        """Add `self` with `other`."""
        return _binop(ops.Add, self, other)

    add = radd = __radd__ = __add__

    def __sub__(self, other: NumericValue) -> NumericValue:
        """Subtract `other` from `self`."""
        return _binop(ops.Subtract, self, other)

    sub = __sub__

    def __rsub__(self, other: NumericValue) -> NumericValue:
        """Subtract `self` from `other`."""
        return _binop(ops.Subtract, other, self)

    rsub = __rsub__

    def __mul__(self, other: NumericValue) -> NumericValue:
        """Multiply `self` and `other`."""
        return _binop(ops.Multiply, self, other)

    mul = rmul = __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide `self` by `other`."""
        return _binop(ops.Divide, self, other)

    div = __div__ = __truediv__

    def __rtruediv__(self, other: NumericValue) -> NumericValue:
        """Divide `other` by `self`."""
        return _binop(ops.Divide, other, self)

    rdiv = __rdiv__ = __rtruediv__

    def __floordiv__(
        self,
        other: NumericValue,
    ) -> NumericValue:
        """Floor divide `self` by `other`."""
        return _binop(ops.FloorDivide, self, other)

    floordiv = __floordiv__

    def __rfloordiv__(
        self,
        other: NumericValue,
    ) -> NumericValue:
        """Floor divide `other` by `self`."""
        return _binop(ops.FloorDivide, other, self)

    rfloordiv = __rfloordiv__

    def __pow__(self, other: NumericValue) -> NumericValue:
        """Raise `self` to the `other`th power."""
        return _binop(ops.Power, self, other)

    pow = __pow__

    def __rpow__(self, other: NumericValue) -> NumericValue:
        """Raise `other` to the `self`th power."""
        return _binop(ops.Power, other, self)

    rpow = __rpow__

    def __mod__(self, other: NumericValue) -> NumericValue:
        """Compute `self` modulo `other`."""
        return _binop(ops.Modulus, self, other)

    mod = __mod__

    def __rmod__(self, other: NumericValue) -> NumericValue:
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
    def median(self, where: ir.BooleanValue | None = None) -> NumericScalar:
        """Return the median of the column.

        Parameters
        ----------
        where
            Optional boolean expression. If given, only the values where
            `where` evaluates to true will be considered for the median.

        Returns
        -------
        NumericScalar
            Median of the column
        """
        return ops.Median(self, where=self._bind_reduction_filter(where)).to_expr()

    def quantile(
        self,
        quantile: Sequence[NumericValue | float],
        where: ir.BooleanValue | None = None,
    ) -> NumericScalar:
        """Return value at the given quantile.

        Parameters
        ----------
        quantile
            `0 <= quantile <= 1`, the quantile(s) to compute
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
        return op(self, quantile, where=self._bind_reduction_filter(where)).to_expr()

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
        return ops.StandardDev(
            self, how=how, where=self._bind_reduction_filter(where)
        ).to_expr()

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
        return ops.Variance(
            self, how=how, where=self._bind_reduction_filter(where)
        ).to_expr()

    def corr(
        self,
        right: NumericColumn,
        where: ir.BooleanValue | None = None,
        how: Literal["sample", "pop"] = "sample",
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
        return ops.Correlation(
            self, right, how=how, where=self._bind_reduction_filter(where)
        ).to_expr()

    def cov(
        self,
        right: NumericColumn,
        where: ir.BooleanValue | None = None,
        how: Literal["sample", "pop"] = "sample",
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
        return ops.Covariance(
            self, right, how=how, where=self._bind_reduction_filter(where)
        ).to_expr()

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
        return ops.Mean(self, where=self._bind_reduction_filter(where)).to_expr()

    def cummean(self, *, where=None, group_by=None, order_by=None) -> NumericColumn:
        """Return the cumulative mean of the input."""
        return self.mean(where=where).over(
            ibis.cumulative_window(group_by=group_by, order_by=order_by)
        )

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
        return ops.Sum(self, where=self._bind_reduction_filter(where)).to_expr()

    def cumsum(self, *, where=None, group_by=None, order_by=None) -> NumericColumn:
        """Return the cumulative sum of the input."""
        return self.sum(where=where).over(
            ibis.cumulative_window(group_by=group_by, order_by=order_by)
        )

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
            if nbins is None:
                raise ValueError("`nbins` is required if `binwidth` is not provided")

            if base is None:
                base = self.min() - eps

            binwidth = (self.max() - base) / nbins

        return ((self - base) / binwidth).floor()


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

    def __and__(self, other: IntegerValue) -> IntegerValue:
        """Bitwise and `self` with `other`."""
        return _binop(ops.BitwiseAnd, self, other)

    __rand__ = __and__

    def __or__(self, other: IntegerValue) -> IntegerValue:
        """Bitwise or `self` with `other`."""
        return _binop(ops.BitwiseOr, self, other)

    __ror__ = __or__

    def __xor__(self, other: IntegerValue) -> IntegerValue:
        """Bitwise xor `self` with `other`."""
        return _binop(ops.BitwiseXor, self, other)

    __rxor__ = __xor__

    def __lshift__(self, other: IntegerValue) -> IntegerValue:
        """Bitwise left shift `self` with `other`."""
        return _binop(ops.BitwiseLeftShift, self, other)

    def __rlshift__(self, other: IntegerValue) -> IntegerValue:
        """Bitwise left shift `self` with `other`."""
        return _binop(ops.BitwiseLeftShift, other, self)

    def __rshift__(self, other: IntegerValue) -> IntegerValue:
        """Bitwise right shift `self` with `other`."""
        return _binop(ops.BitwiseRightShift, self, other)

    def __rrshift__(self, other: IntegerValue) -> IntegerValue:
        """Bitwise right shift `self` with `other`."""
        return _binop(ops.BitwiseRightShift, other, self)

    def __invert__(self) -> IntegerValue:
        """Bitwise not of `self`.

        Returns
        -------
        IntegerValue
            Inverted bits of `self`.
        """
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
        return ops.BitAnd(self, where=self._bind_reduction_filter(where)).to_expr()

    def bit_or(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise or operator."""
        return ops.BitOr(self, where=self._bind_reduction_filter(where)).to_expr()

    def bit_xor(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise exclusive or operator."""
        return ops.BitXor(self, where=self._bind_reduction_filter(where)).to_expr()


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
