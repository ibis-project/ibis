"""Operations for numeric expressions."""

from __future__ import annotations

import operator
from typing import Optional

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis import util
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Binary, Unary, Value

Integer = Value[dt.Integer]
SoftNumeric = Value[dt.Numeric | dt.Boolean]
StrictNumeric = Value[dt.Numeric]


@public
class NumericBinary(Binary):
    left: SoftNumeric
    right: SoftNumeric


@public
class Add(NumericBinary):
    """Add two values."""

    dtype = rlz.numeric_like("args", operator.add)


@public
class Multiply(NumericBinary):
    """Multiply two values."""

    dtype = rlz.numeric_like("args", operator.mul)


@public
class Power(NumericBinary):
    """Raise the left value to the power of the right value."""

    @property
    def dtype(self):
        dtypes = (arg.dtype for arg in self.args)
        if util.all_of(dtypes, dt.Integer):
            return dt.float64
        else:
            return rlz.highest_precedence_dtype(self.args)


@public
class Subtract(NumericBinary):
    """Subtract the right value from the left value."""

    dtype = rlz.numeric_like("args", operator.sub)


@public
class Divide(NumericBinary):
    """Divide the left value by the right value."""

    dtype = dt.float64


@public
class FloorDivide(Divide):
    """Divide the left value by the right value and round down to the nearest integer."""

    dtype = dt.int64


@public
class Modulus(NumericBinary):
    """Return the remainder after the division of the left value by the right value."""

    dtype = rlz.numeric_like("args", operator.mod)


@public
class Negate(Unary):
    """Negate the value."""

    arg: Value[dt.Numeric | dt.Interval]

    dtype = rlz.dtype_like("arg")


@public
class IsNan(Unary):
    """Check if the value is NaN."""

    arg: Value[dt.Floating]

    dtype = dt.boolean


@public
class IsInf(Unary):
    """Check if the value is infinite."""

    arg: Value[dt.Floating]

    dtype = dt.boolean


@public
class Abs(Unary):
    """Absolute value."""

    arg: SoftNumeric

    dtype = rlz.dtype_like("arg")


@public
class Ceil(Unary):
    """Round up to the nearest integer value greater than or equal to this value."""

    arg: SoftNumeric

    @property
    def dtype(self):
        if self.arg.dtype.is_decimal():
            return self.arg.dtype
        else:
            return dt.int64


@public
class Floor(Unary):
    """Round down to the nearest integer value less than or equal to this value."""

    arg: SoftNumeric

    @property
    def dtype(self):
        if self.arg.dtype.is_decimal():
            return self.arg.dtype
        else:
            return dt.int64


@public
class Round(Value):
    """Round a value."""

    arg: StrictNumeric
    digits: Integer

    shape = rlz.shape_like("arg")

    @property
    def dtype(self):
        digits = self.digits
        arg_dtype = self.arg.dtype

        raw_digits = getattr(digits, "value", None)

        # decimals with literal-typed digits return decimals
        if arg_dtype.is_decimal() and raw_digits is not None:
            return arg_dtype.copy(scale=raw_digits)

        nullable = arg_dtype.nullable

        # if digits are unspecified that means round to an integer
        if raw_digits is not None and raw_digits == 0:
            return dt.int64.copy(nullable=nullable)

        # otherwise one of the following is true:
        # 1. digits are specified as a more complex expression
        # 2. self.arg is a double column
        return dt.double.copy(nullable=nullable)


@public
class Clip(Value):
    """Clip a value to a specified range."""

    arg: StrictNumeric
    lower: Optional[StrictNumeric] = None
    upper: Optional[StrictNumeric] = None

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("arg")


@public
class BaseConvert(Value):
    """Convert a number from one base to another."""

    # TODO(kszucs): this should be Integer simply
    arg: Value[dt.Integer | dt.String]
    from_base: Integer
    to_base: Integer

    dtype = dt.string
    shape = rlz.shape_like("args")


@public
class MathUnary(Unary):
    """Base class for unary math operations."""

    arg: SoftNumeric

    @attribute
    def dtype(self):
        return dt.higher_precedence(self.arg.dtype, dt.float64)


class ExpandingMathUnary(MathUnary):
    @attribute
    def dtype(self):
        if self.arg.dtype.is_decimal():
            return self.arg.dtype
        else:
            return dt.float64


@public
class Exp(ExpandingMathUnary):
    """Exponential function."""


@public
class Sign(Unary):
    """Sign of the value."""

    arg: SoftNumeric

    dtype = rlz.dtype_like("arg")


@public
class Sqrt(MathUnary):
    """Square root of the value."""


@public
class Logarithm(MathUnary):
    """Base class for logarithmic operations."""

    arg: StrictNumeric


@public
class Log(Logarithm):
    """Logarithm with a specific base."""

    base: Optional[StrictNumeric] = None


@public
class Ln(Logarithm):
    """Natural logarithm."""


@public
class Log2(Logarithm):
    """Logarithm base 2."""


@public
class Log10(Logarithm):
    """Logarithm base 10."""


@public
class Degrees(ExpandingMathUnary):
    """Converts radians to degrees."""


@public
class Radians(MathUnary):
    """Converts degrees to radians."""


@public
class TrigonometricUnary(MathUnary):
    """Trigonometric base unary."""


@public
class TrigonometricBinary(Binary):
    """Trigonometric base binary."""

    left: SoftNumeric
    right: SoftNumeric

    dtype = dt.float64


@public
class Acos(TrigonometricUnary):
    """Returns the arc cosine of x."""


@public
class Asin(TrigonometricUnary):
    """Returns the arc sine of x."""


@public
class Atan(TrigonometricUnary):
    """Returns the arc tangent of x."""


@public
class Atan2(TrigonometricBinary):
    """Returns the arc tangent of x and y."""


@public
class Cos(TrigonometricUnary):
    """Returns the cosine of x."""


@public
class Cot(TrigonometricUnary):
    """Returns the cotangent of x."""


@public
class Sin(TrigonometricUnary):
    """Returns the sine of x."""


@public
class Tan(TrigonometricUnary):
    """Returns the tangent of x."""


@public
class BitwiseNot(Unary):
    """Bitwise NOT operation."""

    arg: Integer

    dtype = rlz.numeric_like("args", operator.invert)


@public
class BitwiseBinary(Binary):
    """Base class for bitwise binary operations."""

    left: Integer
    right: Integer


@public
class BitwiseAnd(BitwiseBinary):
    """Bitwise AND operation."""

    dtype = rlz.numeric_like("args", operator.and_)


@public
class BitwiseOr(BitwiseBinary):
    """Bitwise OR operation."""

    dtype = rlz.numeric_like("args", operator.or_)


@public
class BitwiseXor(BitwiseBinary):
    """Bitwise XOR operation."""

    dtype = rlz.numeric_like("args", operator.xor)


@public
class BitwiseLeftShift(BitwiseBinary):
    """Bitwise left shift operation."""

    shape = rlz.shape_like("args")
    dtype = dt.int64


@public
class BitwiseRightShift(BitwiseBinary):
    """Bitwise right shift operation."""

    shape = rlz.shape_like("args")
    dtype = dt.int64
