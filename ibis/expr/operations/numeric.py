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
    dtype = rlz.numeric_like("args", operator.add)


@public
class Multiply(NumericBinary):
    dtype = rlz.numeric_like("args", operator.mul)


@public
class Power(NumericBinary):
    @property
    def dtype(self):
        dtypes = (arg.dtype for arg in self.args)
        if util.all_of(dtypes, dt.Integer):
            return dt.float64
        else:
            return rlz.highest_precedence_dtype(self.args)


@public
class Subtract(NumericBinary):
    dtype = rlz.numeric_like("args", operator.sub)


@public
class Divide(NumericBinary):
    dtype = dt.float64


@public
class FloorDivide(Divide):
    dtype = dt.int64


@public
class Modulus(NumericBinary):
    dtype = rlz.numeric_like("args", operator.mod)


@public
class Negate(Unary):
    arg: Value[dt.Numeric | dt.Interval]

    dtype = rlz.dtype_like("arg")


@public
class IsNan(Unary):
    arg: Value[dt.Floating]

    dtype = dt.boolean


@public
class IsInf(Unary):
    arg: Value[dt.Floating]

    dtype = dt.boolean


@public
class Abs(Unary):
    """Absolute value."""

    arg: SoftNumeric

    dtype = rlz.dtype_like("arg")


@public
class Ceil(Unary):
    """Round up to the nearest integer value greater than or equal to this value.

    Returns
    -------
    DecimalValue | IntegerValue
        Decimal values: yield decimal
        Other numeric values: yield integer (int32)
    """

    arg: SoftNumeric

    @property
    def dtype(self):
        if self.arg.dtype.is_decimal():
            return self.arg.dtype
        else:
            return dt.int64


@public
class Floor(Unary):
    """Round down to the nearest integer value less than or equal to this value.

    Returns
    -------
    DecimalValue | IntegerValue
        Decimal values: yield decimal
        Other numeric values: yield integer (int32)
    """

    arg: SoftNumeric

    @property
    def dtype(self):
        if self.arg.dtype.is_decimal():
            return self.arg.dtype
        else:
            return dt.int64


@public
class Round(Value):
    arg: StrictNumeric
    # TODO(kszucs): the default should be 0 instead of being None
    digits: Optional[Integer] = None

    shape = rlz.shape_like("arg")

    @property
    def dtype(self):
        if self.arg.dtype.is_decimal():
            return self.arg.dtype
        elif self.digits is None:
            return dt.int64
        else:
            return dt.double


@public
class Clip(Value):
    arg: StrictNumeric
    lower: Optional[StrictNumeric] = None
    upper: Optional[StrictNumeric] = None

    dtype = rlz.dtype_like("arg")
    shape = rlz.shape_like("arg")


@public
class BaseConvert(Value):
    # TODO(kszucs): this should be Integer simply
    arg: Value[dt.Integer | dt.String]
    from_base: Integer
    to_base: Integer

    dtype = dt.string
    shape = rlz.shape_like("args")


@public
class MathUnary(Unary):
    arg: SoftNumeric

    @attribute
    def dtype(self):
        return dt.higher_precedence(self.arg.dtype, dt.float64)


@public
class ExpandingMathUnary(MathUnary):
    @attribute
    def dtype(self):
        if self.arg.dtype.is_decimal():
            return self.arg.dtype
        else:
            return dt.float64


@public
class Exp(ExpandingMathUnary):
    pass


@public
class Sign(Unary):
    arg: SoftNumeric

    dtype = rlz.dtype_like("arg")


@public
class Sqrt(MathUnary):
    pass


@public
class Logarithm(MathUnary):
    arg: StrictNumeric


@public
class Log(Logarithm):
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


# TRIGONOMETRIC OPERATIONS


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
    arg: Integer

    dtype = rlz.numeric_like("args", operator.invert)


@public
class BitwiseBinary(Binary):
    left: Integer
    right: Integer


@public
class BitwiseAnd(BitwiseBinary):
    dtype = rlz.numeric_like("args", operator.and_)


@public
class BitwiseOr(BitwiseBinary):
    dtype = rlz.numeric_like("args", operator.or_)


@public
class BitwiseXor(BitwiseBinary):
    dtype = rlz.numeric_like("args", operator.xor)


@public
class BitwiseLeftShift(BitwiseBinary):
    shape = rlz.shape_like("args")
    dtype = dt.int64


@public
class BitwiseRightShift(BitwiseBinary):
    shape = rlz.shape_like("args")
    dtype = dt.int64
