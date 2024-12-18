from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from public import public

import ibis
import ibis.expr.operations as ops
from ibis.common.exceptions import IbisTypeError
from ibis.expr.types.core import _binop
from ibis.expr.types.generic import Column, Scalar, Value
from ibis.util import deprecated

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class NumericValue(Value):
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
        return ops.Negate(self).to_expr()

    def __neg__(self) -> NumericValue:
        """Negate `self`.

        Returns
        -------
        NumericValue
            `self` negated
        """
        return self.negate()

    def round(self, digits: int | IntegerValue = 0) -> NumericValue:
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
        ┏━━━━━━━━━━━━━━━━━━┓
        ┃ Round(values, 0) ┃
        ┡━━━━━━━━━━━━━━━━━━┩
        │ int64            │
        ├──────────────────┤
        │                1 │
        │                2 │
        │                2 │
        │                3 │
        └──────────────────┘
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
        >>> t = ibis.memtable({"values": [-1, -2, 3]})
        >>> t.values.cot()
        ┏━━━━━━━━━━━━━┓
        ┃ Cot(values) ┃
        ┡━━━━━━━━━━━━━┩
        │ float64     │
        ├─────────────┤
        │   -0.642093 │
        │    0.457658 │
        │   -7.015253 │
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.zones.fetch()
        >>> t.x_cent.point(t.y_cent)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ GeoPoint(x_cent, y_cent)         ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ point:geometry                   │
        ├──────────────────────────────────┤
        │ <POINT (935996.821 191376.75)>   │
        │ <POINT (1031085.719 164018.754)> │
        │ <POINT (1026452.617 254265.479)> │
        │ <POINT (990633.981 202959.782)>  │
        │ <POINT (931871.37 140681.351)>   │
        │ <POINT (964319.735 157998.936)>  │
        │ <POINT (1006496.679 216719.218)> │
        │ <POINT (1005551.571 222936.088)> │
        │ <POINT (1043002.677 212969.849)> │
        │ <POINT (1042223.605 186706.496)> │
        │ …                                │
        └──────────────────────────────────┘
        """
        return ops.GeoPoint(self, right).to_expr()


@public
class NumericScalar(Scalar, NumericValue):
    pass


@public
class NumericColumn(Column, NumericValue):
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "values": [1, 3, 3, 4, 5, 7],
        ...     }
        ... )
        >>> t.values.std()
        ┌──────────┐
        │ 2.041241 │
        └──────────┘
        >>> t.mutate(std_col=t.values.std())
        ┏━━━━━━━━┳━━━━━━━━━━┓
        ┃ values ┃ std_col  ┃
        ┡━━━━━━━━╇━━━━━━━━━━┩
        │ int64  │ float64  │
        ├────────┼──────────┤
        │      1 │ 2.041241 │
        │      3 │ 2.041241 │
        │      3 │ 2.041241 │
        │      4 │ 2.041241 │
        │      5 │ 2.041241 │
        │      7 │ 2.041241 │
        └────────┴──────────┘
        """
        return ops.StandardDev(
            self, how=how, where=self._bind_to_parent_table(where)
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "values": [1, 3, 3, 4, 5, 7],
        ...     }
        ... )
        >>> t.values.var()
        ┌──────────┐
        │ 4.166667 │
        └──────────┘
        >>> t.mutate(var_col=t.values.var())
        ┏━━━━━━━━┳━━━━━━━━━━┓
        ┃ values ┃ var_col  ┃
        ┡━━━━━━━━╇━━━━━━━━━━┩
        │ int64  │ float64  │
        ├────────┼──────────┤
        │      1 │ 4.166667 │
        │      3 │ 4.166667 │
        │      3 │ 4.166667 │
        │      4 │ 4.166667 │
        │      5 │ 4.166667 │
        │      7 │ 4.166667 │
        └────────┴──────────┘
        """
        return ops.Variance(
            self, how=how, where=self._bind_to_parent_table(where)
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "left": [1, 3, 3, 4, 5, 7],
        ...         "right": [7, 5, 4, 3, 3, 1],
        ...     }
        ... )
        >>> t.left.corr(t.right, how="pop")
        ┌────────┐
        │ -0.968 │
        └────────┘
        >>> t.mutate(corr_col=t.left.corr(t.right, how="pop"))
        ┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
        ┃ left  ┃ right ┃ corr_col ┃
        ┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
        │ int64 │ int64 │ float64  │
        ├───────┼───────┼──────────┤
        │     1 │     7 │   -0.968 │
        │     3 │     5 │   -0.968 │
        │     3 │     4 │   -0.968 │
        │     4 │     3 │   -0.968 │
        │     5 │     3 │   -0.968 │
        │     7 │     1 │   -0.968 │
        └───────┴───────┴──────────┘
        """
        return ops.Correlation(
            self,
            self._bind_to_parent_table(right),
            how=how,
            where=self._bind_to_parent_table(where),
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "left": [1, 3, 3, 4, 5, 7],
        ...         "right": [7, 5, 4, 3, 3, 1],
        ...     }
        ... )
        >>> t.left.cov(t.right)
        ┌───────────┐
        │ -4.033333 │
        └───────────┘
        >>> t.left.cov(t.right, how="pop")
        ┌───────────┐
        │ -3.361111 │
        └───────────┘
        >>> t.mutate(cov_col=t.left.cov(t.right, how="pop"))
        ┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
        ┃ left  ┃ right ┃ cov_col   ┃
        ┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
        │ int64 │ int64 │ float64   │
        ├───────┼───────┼───────────┤
        │     1 │     7 │ -3.361111 │
        │     3 │     5 │ -3.361111 │
        │     3 │     4 │ -3.361111 │
        │     4 │     3 │ -3.361111 │
        │     5 │     3 │ -3.361111 │
        │     7 │     1 │ -3.361111 │
        └───────┴───────┴───────────┘
        """
        return ops.Covariance(
            self,
            self._bind_to_parent_table(right),
            how=how,
            where=self._bind_to_parent_table(where),
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "id": [1, 2, 3, 4, 5, 6],
        ...         "grouper": ["a", "a", "a", "b", "b", "c"],
        ...         "values": [3, 2, 1, 2, 3, 2],
        ...     }
        ... )
        >>> t.mutate(mean_col=t.values.mean())
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
        ┃ id    ┃ grouper ┃ values ┃ mean_col ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
        │ int64 │ string  │ int64  │ float64  │
        ├───────┼─────────┼────────┼──────────┤
        │     1 │ a       │      3 │ 2.166667 │
        │     2 │ a       │      2 │ 2.166667 │
        │     3 │ a       │      1 │ 2.166667 │
        │     4 │ b       │      2 │ 2.166667 │
        │     5 │ b       │      3 │ 2.166667 │
        │     6 │ c       │      2 │ 2.166667 │
        └───────┴─────────┴────────┴──────────┘

        >>> t.mutate(mean_col=t.values.mean(where=t.grouper != "c"))
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
        ┃ id    ┃ grouper ┃ values ┃ mean_col ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
        │ int64 │ string  │ int64  │ float64  │
        ├───────┼─────────┼────────┼──────────┤
        │     1 │ a       │      3 │      2.2 │
        │     2 │ a       │      2 │      2.2 │
        │     3 │ a       │      1 │      2.2 │
        │     4 │ b       │      2 │      2.2 │
        │     5 │ b       │      3 │      2.2 │
        │     6 │ c       │      2 │      2.2 │
        └───────┴─────────┴────────┴──────────┘
        """
        # TODO(kszucs): remove the alias from the reduction method in favor
        # of default name generated by ops.Value operations
        return ops.Mean(self, where=self._bind_to_parent_table(where)).to_expr()

    def cummean(self, *, where=None, group_by=None, order_by=None) -> NumericColumn:
        """Return the cumulative mean of the input.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "id": [1, 2, 3, 4, 5, 6],
        ...         "grouper": ["a", "a", "a", "b", "b", "c"],
        ...         "values": [3, 2, 1, 2, 3, 2],
        ...     }
        ... )
        >>> t.mutate(cummean=t.values.cummean()).order_by("id")
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
        ┃ id    ┃ grouper ┃ values ┃ cummean  ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
        │ int64 │ string  │ int64  │ float64  │
        ├───────┼─────────┼────────┼──────────┤
        │     1 │ a       │      3 │ 3.000000 │
        │     2 │ a       │      2 │ 2.500000 │
        │     3 │ a       │      1 │ 2.000000 │
        │     4 │ b       │      2 │ 2.000000 │
        │     5 │ b       │      3 │ 2.200000 │
        │     6 │ c       │      2 │ 2.166667 │
        └───────┴─────────┴────────┴──────────┘

        >>> t.mutate(cummean=t.values.cummean(where=t.grouper != "c", group_by="grouper")).order_by(
        ...     "id"
        ... )
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
        ┃ id    ┃ grouper ┃ values ┃ cummean ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
        │ int64 │ string  │ int64  │ float64 │
        ├───────┼─────────┼────────┼─────────┤
        │     1 │ a       │      3 │     3.0 │
        │     2 │ a       │      2 │     2.5 │
        │     3 │ a       │      1 │     2.0 │
        │     4 │ b       │      2 │     2.0 │
        │     5 │ b       │      3 │     2.5 │
        │     6 │ c       │      2 │    NULL │
        └───────┴─────────┴────────┴─────────┘
        """
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "id": [1, 2, 3, 4, 5, 6],
        ...         "grouper": ["a", "a", "a", "b", "b", "c"],
        ...         "values": [3, 2, 1, 2, 3, 2],
        ...     }
        ... )
        >>> t.mutate(sum_col=t.values.sum())
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
        ┃ id    ┃ grouper ┃ values ┃ sum_col ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
        │ int64 │ string  │ int64  │ int64   │
        ├───────┼─────────┼────────┼─────────┤
        │     1 │ a       │      3 │      13 │
        │     2 │ a       │      2 │      13 │
        │     3 │ a       │      1 │      13 │
        │     4 │ b       │      2 │      13 │
        │     5 │ b       │      3 │      13 │
        │     6 │ c       │      2 │      13 │
        └───────┴─────────┴────────┴─────────┘

        >>> t.mutate(sum_col=t.values.sum(where=t.grouper != "c"))
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
        ┃ id    ┃ grouper ┃ values ┃ sum_col ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
        │ int64 │ string  │ int64  │ int64   │
        ├───────┼─────────┼────────┼─────────┤
        │     1 │ a       │      3 │      11 │
        │     2 │ a       │      2 │      11 │
        │     3 │ a       │      1 │      11 │
        │     4 │ b       │      2 │      11 │
        │     5 │ b       │      3 │      11 │
        │     6 │ c       │      2 │      11 │
        └───────┴─────────┴────────┴─────────┘
        """
        return ops.Sum(self, where=self._bind_to_parent_table(where)).to_expr()

    def cumsum(self, *, where=None, group_by=None, order_by=None) -> NumericColumn:
        """Return the cumulative sum of the input.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "id": [1, 2, 3, 4, 5, 6],
        ...         "grouper": ["a", "a", "a", "b", "b", "c"],
        ...         "values": [3, 2, 1, 2, 3, 2],
        ...     }
        ... )
        >>> t.mutate(cumsum=t.values.cumsum())
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ id    ┃ grouper ┃ values ┃ cumsum ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ int64 │ string  │ int64  │ int64  │
        ├───────┼─────────┼────────┼────────┤
        │     1 │ a       │      3 │      3 │
        │     2 │ a       │      2 │      5 │
        │     3 │ a       │      1 │      6 │
        │     4 │ b       │      2 │      8 │
        │     5 │ b       │      3 │     11 │
        │     6 │ c       │      2 │     13 │
        └───────┴─────────┴────────┴────────┘

        >>> t.mutate(cumsum=t.values.cumsum(where=t.grouper != "c"))
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ id    ┃ grouper ┃ values ┃ cumsum ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ int64 │ string  │ int64  │ int64  │
        ├───────┼─────────┼────────┼────────┤
        │     1 │ a       │      3 │      3 │
        │     2 │ a       │      2 │      5 │
        │     3 │ a       │      1 │      6 │
        │     4 │ b       │      2 │      8 │
        │     5 │ b       │      3 │     11 │
        │     6 │ c       │      2 │     11 │
        └───────┴─────────┴────────┴────────┘
        """
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "values": [-1, 3, 5, 6, 8, 10, 11],
        ...     }
        ... )
        >>> buckets = [0, 5, 10]
        >>> t.mutate(
        ...     bucket_closed_left=t.values.bucket(buckets),
        ...     bucket_closed_right=t.values.bucket(buckets, closed="right"),
        ...     bucket_over_under=t.values.bucket(buckets, include_over=True, include_under=True),
        ... )
        ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
        ┃ values ┃ bucket_closed_left ┃ bucket_closed_right ┃ bucket_over_under ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
        │ int64  │ int8               │ int8                │ int8              │
        ├────────┼────────────────────┼─────────────────────┼───────────────────┤
        │     -1 │               NULL │                NULL │                 0 │
        │      3 │                  0 │                   0 │                 1 │
        │      5 │                  1 │                   0 │                 2 │
        │      6 │                  1 │                   1 │                 2 │
        │      8 │                  1 │                   1 │                 2 │
        │     10 │                  1 │                   1 │                 2 │
        │     11 │               NULL │                NULL │                 3 │
        └────────┴────────────────────┴─────────────────────┴───────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "values": [-1, 3, 5, 6, 8, 10, 11, 23, 25],
        ...     }
        ... )

        Compute a histogram with 5 bins.

        >>> t.mutate(histogram=t.values.histogram(nbins=5))
        ┏━━━━━━━━┳━━━━━━━━━━━┓
        ┃ values ┃ histogram ┃
        ┡━━━━━━━━╇━━━━━━━━━━━┩
        │ int64  │ int64     │
        ├────────┼───────────┤
        │     -1 │         0 │
        │      3 │         0 │
        │      5 │         1 │
        │      6 │         1 │
        │      8 │         1 │
        │     10 │         2 │
        │     11 │         2 │
        │     23 │         4 │
        │     25 │         4 │
        └────────┴───────────┘

        Compute a histogram with a fixed bin width of 10.
        >>> t.mutate(histogram=t.values.histogram(binwidth=10))
        ┏━━━━━━━━┳━━━━━━━━━━━┓
        ┃ values ┃ histogram ┃
        ┡━━━━━━━━╇━━━━━━━━━━━┩
        │ int64  │ int64     │
        ├────────┼───────────┤
        │     -1 │         0 │
        │      3 │         0 │
        │      5 │         0 │
        │      6 │         0 │
        │      8 │         0 │
        │     10 │         1 │
        │     11 │         1 │
        │     23 │         2 │
        │     25 │         2 │
        └────────┴───────────┘
        """

        if nbins is not None and binwidth is not None:
            raise ValueError(
                f"Cannot pass both `nbins` (got {nbins}) and `binwidth` (got {binwidth})"
            )

        if base is None:
            base = self.min() - eps

        if binwidth is None:
            if nbins is None:
                raise ValueError("`nbins` is required if `binwidth` is not provided")

            binwidth = (self.max() - base) / nbins

        if nbins is None:
            nbins = ((self.max() - base) / binwidth).ceil()

        return ((self - base) / binwidth).floor().clip(-1, nbins - 1)

    def approx_quantile(
        self,
        quantile: float | ir.NumericValue | Sequence[ir.NumericValue | float],
        where: ir.BooleanValue | None = None,
    ) -> NumericScalar:
        """Compute one or more approximate quantiles of a column.

        ::: {.callout-note}
        ## The result may or may not be exact

        Whether the result is an approximation depends on the backend.
        :::

        Parameters
        ----------
        quantile
            `0 <= quantile <= 1`, or an array of such values
            indicating the quantile or quantiles to compute
        where
            Boolean filter for input values

        Returns
        -------
        Scalar
            Quantile of the input

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()

        Compute the approximate 0.50 quantile of `bill_depth_mm`.

        >>> t.bill_depth_mm.approx_quantile(0.50)
        ┌────────┐
        │ 17.318 │
        └────────┘

        Compute multiple approximate quantiles in one call - in this case the
        result is an array.

        >>> t.bill_depth_mm.approx_quantile([0.25, 0.75])
        ┌────────────────────────┐
        │ [15.565625, 18.671875] │
        └────────────────────────┘
        """
        if isinstance(quantile, Sequence):
            op = ops.ApproxMultiQuantile
        else:
            op = ops.ApproxQuantile
        return op(self, quantile, where=self._bind_to_parent_table(where)).to_expr()


@public
class IntegerValue(NumericValue):
    def as_timestamp(
        self,
        unit: Literal["s", "ms", "us"],
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "int_col": [0, 1730501716, 2147483647],
        ...     }
        ... )
        >>> t.int_col.as_timestamp("s")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ TimestampFromUNIX(int_col, SECOND) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ timestamp                          │
        ├────────────────────────────────────┤
        │ 1970-01-01 00:00:00                │
        │ 2024-11-01 22:55:16                │
        │ 2038-01-19 03:14:07                │
        └────────────────────────────────────┘
        """
        return ops.TimestampFromUNIX(self, unit).to_expr()

    def as_interval(
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

        Examples
        --------
        >>> from datetime import datetime
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "timestamp_col": [
        ...             datetime(2024, 1, 1, 0, 0, 0),
        ...             datetime(2024, 1, 1, 0, 0, 0),
        ...             datetime(2024, 1, 1, 0, 0, 0),
        ...         ],
        ...         "int_col": [1, 2, 3],
        ...     }
        ... )
        >>> t.int_col.as_interval("h")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ IntervalFromInteger(int_col, HOUR)                         ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ interval('h')                                              │
        ├────────────────────────────────────────────────────────────┤
        │ MonthDayNano(months=0, days=0, nanoseconds=3600000000000)  │
        │ MonthDayNano(months=0, days=0, nanoseconds=7200000000000)  │
        │ MonthDayNano(months=0, days=0, nanoseconds=10800000000000) │
        └────────────────────────────────────────────────────────────┘

        >>> t.mutate(timestamp_added_col=t.timestamp_col + t.int_col.as_interval("h"))
        ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
        ┃ timestamp_col       ┃ int_col ┃ timestamp_added_col ┃
        ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
        │ timestamp           │ int64   │ timestamp           │
        ├─────────────────────┼─────────┼─────────────────────┤
        │ 2024-01-01 00:00:00 │       1 │ 2024-01-01 01:00:00 │
        │ 2024-01-01 00:00:00 │       2 │ 2024-01-01 02:00:00 │
        │ 2024-01-01 00:00:00 │       3 │ 2024-01-01 03:00:00 │
        └─────────────────────┴─────────┴─────────────────────┘
        """
        return ops.IntervalFromInteger(self, unit).to_expr()

    @deprecated(as_of="10.0", instead="use as_timestamp() instead")
    def to_timestamp(
        self,
        unit: Literal["s", "ms", "us"] = "s",
    ) -> ir.TimestampValue:
        return self.as_timestamp(unit=unit)

    @deprecated(as_of="10.0", instead="use as_interval() instead")
    def to_interval(
        self,
        unit: Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"] = "s",
    ) -> ir.IntervalValue:
        return self.as_interval(unit=unit)

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


@public
class IntegerScalar(NumericScalar, IntegerValue):
    pass


@public
class IntegerColumn(NumericColumn, IntegerValue):
    def bit_and(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise and operator.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"x": [-1, 0, 1]})
        >>> t.x.bit_and()
        ┌───┐
        │ 0 │
        └───┘
        >>> t.x.bit_and(where=t.x != 0)
        ┌───┐
        │ 1 │
        └───┘
        """
        return ops.BitAnd(self, where=self._bind_to_parent_table(where)).to_expr()

    def bit_or(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise or operator.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"x": [-1, 0, 1]})
        >>> t.x.bit_or()
        ┌────┐
        │ -1 │
        └────┘
        >>> t.x.bit_or(where=t.x >= 0)
        ┌───┐
        │ 1 │
        └───┘
        """
        return ops.BitOr(self, where=self._bind_to_parent_table(where)).to_expr()

    def bit_xor(self, where: ir.BooleanValue | None = None) -> IntegerScalar:
        """Aggregate the column using the bitwise exclusive or operator.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"x": [-1, 0, 1]})
        >>> t.x.bit_xor()
        ┌────┐
        │ -2 │
        └────┘
        >>> t.x.bit_xor(where=t.x >= 0)
        ┌───┐
        │ 1 │
        └───┘
        """
        return ops.BitXor(self, where=self._bind_to_parent_table(where)).to_expr()


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
