from __future__ import annotations

import operator

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis import util
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Binary, Unary, Value


@public
class NumericBinary(Binary):
    left = rlz.numeric
    right = rlz.numeric


@public
class Add(NumericBinary):
    output_dtype = rlz.numeric_like("args", operator.add)


@public
class Multiply(NumericBinary):
    output_dtype = rlz.numeric_like("args", operator.mul)


@public
class Power(NumericBinary):
    @property
    def output_dtype(self):
        dtypes = (arg.output_dtype for arg in self.args)
        if util.all_of(dtypes, dt.Integer):
            return dt.float64
        else:
            return rlz.highest_precedence_dtype(self.args)


@public
class Subtract(NumericBinary):
    output_dtype = rlz.numeric_like("args", operator.sub)


@public
class Divide(NumericBinary):
    output_dtype = dt.float64


@public
class FloorDivide(Divide):
    output_dtype = dt.int64


@public
class Modulus(NumericBinary):
    output_dtype = rlz.numeric_like("args", operator.mod)


@public
class Negate(Unary):
    arg = rlz.one_of((rlz.numeric, rlz.interval))

    output_dtype = rlz.dtype_like("arg")


@public
class NullIfZero(Unary):
    """Set values to NULL if they are equal to zero.

    Commonly used in cases where divide-by-zero would produce an overflow or
    infinity.

    Equivalent to

    ```python
    (value == 0).ifelse(ibis.NA, value)
    ```

    Returns
    -------
    NumericValue
        The input if not zero otherwise `NULL`.
    """

    arg = rlz.numeric
    output_dtype = rlz.dtype_like("arg")


@public
class IsNan(Unary):
    arg = rlz.floating
    output_dtype = dt.boolean


@public
class IsInf(Unary):
    arg = rlz.floating
    output_dtype = dt.boolean


@public
class Abs(Unary):
    """Absolute value."""

    arg = rlz.numeric
    output_dtype = rlz.dtype_like("arg")


@public
class Ceil(Unary):
    """Round up to the nearest integer value greater than or equal to this value.

    Returns
    -------
    DecimalValue | IntegerValue
        Decimal values: yield decimal
        Other numeric values: yield integer (int32)
    """

    arg = rlz.numeric

    @property
    def output_dtype(self):
        if self.arg.output_dtype.is_decimal():
            return self.arg.output_dtype
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

    arg = rlz.numeric

    @property
    def output_dtype(self):
        if self.arg.output_dtype.is_decimal():
            return self.arg.output_dtype
        else:
            return dt.int64


@public
class Round(Value):
    arg = rlz.numeric
    # TODO(kszucs): the default should be 0 instead of being None
    digits = rlz.optional(rlz.numeric)

    output_shape = rlz.shape_like("arg")

    @property
    def output_dtype(self):
        if self.arg.output_dtype.is_decimal():
            return self.arg.output_dtype
        elif self.digits is None:
            return dt.int64
        else:
            return dt.double


@public
class Clip(Value):
    arg = rlz.strict_numeric
    lower = rlz.optional(rlz.strict_numeric)
    upper = rlz.optional(rlz.strict_numeric)

    output_dtype = rlz.dtype_like("arg")
    output_shape = rlz.shape_like("arg")


@public
class BaseConvert(Value):
    arg = rlz.one_of([rlz.integer, rlz.string])
    from_base = rlz.integer
    to_base = rlz.integer

    output_dtype = dt.string
    output_shape = rlz.shape_like("args")


@public
class MathUnary(Unary):
    arg = rlz.numeric

    @attribute.default
    def output_dtype(self):
        return dt.higher_precedence(self.arg.output_dtype, dt.double)


@public
class ExpandingMathUnary(MathUnary):
    @attribute.default
    def output_dtype(self):
        return dt.higher_precedence(self.arg.output_dtype.largest, dt.double)


@public
class Exp(ExpandingMathUnary):
    pass


@public
class Sign(Unary):
    arg = rlz.numeric
    output_dtype = rlz.dtype_like("arg")


@public
class Sqrt(MathUnary):
    pass


@public
class Logarithm(MathUnary):
    arg = rlz.strict_numeric


@public
class Log(Logarithm):
    arg = rlz.strict_numeric
    base = rlz.optional(rlz.strict_numeric)


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

    left = rlz.numeric
    right = rlz.numeric
    output_dtype = dt.float64


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
    arg = rlz.integer
    output_dtype = rlz.numeric_like("args", operator.invert)


@public
class BitwiseBinary(Binary):
    left = rlz.integer
    right = rlz.integer


@public
class BitwiseAnd(BitwiseBinary):
    output_dtype = rlz.numeric_like("args", operator.and_)


@public
class BitwiseOr(BitwiseBinary):
    output_dtype = rlz.numeric_like("args", operator.or_)


@public
class BitwiseXor(BitwiseBinary):
    output_dtype = rlz.numeric_like("args", operator.xor)


@public
class BitwiseLeftShift(BitwiseBinary):
    output_shape = rlz.shape_like("args")
    output_dtype = dt.int64


@public
class BitwiseRightShift(BitwiseBinary):
    output_shape = rlz.shape_like("args")
    output_dtype = dt.int64
