import operator

from public import public

from ... import util
from .. import datatypes as dt
from .. import rules as rlz
from .. import types as ir
from .core import BinaryOp, UnaryOp, ValueOp


@public
class NumericBinaryOp(BinaryOp):
    left = rlz.numeric
    right = rlz.numeric


@public
class Add(NumericBinaryOp):
    output_type = rlz.numeric_like('args', operator.add)


@public
class Multiply(NumericBinaryOp):
    output_type = rlz.numeric_like('args', operator.mul)


@public
class Power(NumericBinaryOp):
    def output_type(self):
        if util.all_of(self.args, ir.IntegerValue):
            return rlz.shape_like(self.args, dt.float64)
        else:
            return rlz.shape_like(self.args)


@public
class Subtract(NumericBinaryOp):
    output_type = rlz.numeric_like('args', operator.sub)


@public
class Divide(NumericBinaryOp):
    output_type = rlz.shape_like('args', dt.float64)


@public
class FloorDivide(Divide):
    output_type = rlz.shape_like('args', dt.int64)


@public
class Modulus(NumericBinaryOp):
    output_type = rlz.numeric_like('args', operator.mod)


@public
class Negate(UnaryOp):
    arg = rlz.one_of((rlz.numeric, rlz.interval))
    output_type = rlz.typeof('arg')


@public
class NullIfZero(ValueOp):
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
    output_type = rlz.typeof('arg')


@public
class IsNan(ValueOp):
    arg = rlz.floating
    output_type = rlz.shape_like('arg', dt.boolean)


@public
class IsInf(ValueOp):
    arg = rlz.floating
    output_type = rlz.shape_like('arg', dt.boolean)


@public
class Abs(UnaryOp):
    """Absolute value"""

    output_type = rlz.typeof('arg')


@public
class Ceil(UnaryOp):

    """
    Round up to the nearest integer value greater than or equal to this value

    Returns
    -------
    ceiled : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """

    arg = rlz.numeric

    def output_type(self):
        if isinstance(self.arg.type(), dt.Decimal):
            return self.arg._factory
        return rlz.shape_like(self.arg, dt.int64)


@public
class Floor(UnaryOp):

    """
    Round down to the nearest integer value less than or equal to this value

    Returns
    -------
    floored : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """

    arg = rlz.numeric

    def output_type(self):
        if isinstance(self.arg.type(), dt.Decimal):
            return self.arg._factory
        return rlz.shape_like(self.arg, dt.int64)


@public
class Round(ValueOp):
    arg = rlz.numeric
    digits = rlz.optional(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg._factory
        elif self.digits is None:
            return rlz.shape_like(self.arg, dt.int64)
        else:
            return rlz.shape_like(self.arg, dt.double)


@public
class Clip(ValueOp):
    arg = rlz.strict_numeric
    lower = rlz.optional(rlz.strict_numeric)
    upper = rlz.optional(rlz.strict_numeric)
    output_type = rlz.typeof('arg')


@public
class ClipLower(ValueOp):
    arg = rlz.strict_numeric
    lower = rlz.optional(rlz.strict_numeric)
    output_type = rlz.typeof('arg')


@public
class ClipUpper(ValueOp):
    arg = rlz.strict_numeric
    upper = rlz.optional(rlz.strict_numeric)
    output_type = rlz.typeof('arg')


@public
class BaseConvert(ValueOp):
    arg = rlz.one_of([rlz.integer, rlz.string])
    from_base = rlz.integer
    to_base = rlz.integer

    def output_type(self):
        return rlz.shape_like(tuple(self.flat_args()), dt.string)


@public
class MathUnaryOp(UnaryOp):
    arg = rlz.numeric

    def output_type(self):
        arg = self.arg
        if isinstance(self.arg, ir.DecimalValue):
            dtype = arg.type()
        else:
            dtype = dt.double
        return rlz.shape_like(arg, dtype)


@public
class ExpandingTypeMathUnaryOp(MathUnaryOp):
    def output_type(self):
        if not isinstance(self.arg, ir.DecimalValue):
            return super().output_type()
        arg = self.arg
        return rlz.shape_like(arg, arg.type().largest)


@public
class Exp(ExpandingTypeMathUnaryOp):
    pass


@public
class Sign(UnaryOp):
    arg = rlz.numeric
    output_type = rlz.typeof('arg')


@public
class Sqrt(MathUnaryOp):
    pass


@public
class Logarithm(MathUnaryOp):
    arg = rlz.strict_numeric


@public
class Log(Logarithm):
    arg = rlz.strict_numeric
    base = rlz.optional(rlz.strict_numeric)


@public
class Ln(Logarithm):
    """Natural logarithm"""


@public
class Log2(Logarithm):
    """Logarithm base 2"""


@public
class Log10(Logarithm):
    """Logarithm base 10"""


@public
class Degrees(ExpandingTypeMathUnaryOp):
    """Converts radians to degrees"""

    arg = rlz.numeric


@public
class Radians(MathUnaryOp):
    """Converts degrees to radians"""

    arg = rlz.numeric


# TRIGONOMETRIC OPERATIONS


@public
class TrigonometricUnary(MathUnaryOp):
    """Trigonometric base unary"""

    arg = rlz.numeric


@public
class TrigonometricBinary(BinaryOp):
    """Trigonometric base binary"""

    left = rlz.numeric
    right = rlz.numeric
    output_type = rlz.shape_like('args', dt.float64)


@public
class Acos(TrigonometricUnary):
    """Returns the arc cosine of x"""


@public
class Asin(TrigonometricUnary):
    """Returns the arc sine of x"""


@public
class Atan(TrigonometricUnary):
    """Returns the arc tangent of x"""


@public
class Atan2(TrigonometricBinary):
    """Returns the arc tangent of x and y"""


@public
class Cos(TrigonometricUnary):
    """Returns the cosine of x"""


@public
class Cot(TrigonometricUnary):
    """Returns the cotangent of x"""


@public
class Sin(TrigonometricUnary):
    """Returns the sine of x"""


@public
class Tan(TrigonometricUnary):
    """Returns the tangent of x"""
