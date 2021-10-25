from contextlib import suppress

from public import public

from ...common import exceptions as com
from .. import datatypes as dt
from .. import rules as rlz
from ..signature import Argument as Arg
from .core import BinaryOp, UnaryOp, ValueOp
from .generic import BooleanValueOp


@public
class LogicalBinaryOp(BinaryOp):
    left = Arg(rlz.boolean)
    right = Arg(rlz.boolean)
    output_type = rlz.shape_like('args', dt.boolean)


@public
class Not(UnaryOp):
    arg = Arg(rlz.boolean)
    output_type = rlz.shape_like('arg', dt.boolean)


@public
class And(LogicalBinaryOp):
    pass


@public
class Or(LogicalBinaryOp):
    pass


@public
class Xor(LogicalBinaryOp):
    pass


@public
class Comparison(BinaryOp, BooleanValueOp):
    left = Arg(rlz.any)
    right = Arg(rlz.any)

    def __init__(self, left, right):
        """
        Casting rules for type promotions (for resolving the output type) may
        depend in some cases on the target backend.

        TODO: how will overflows be handled? Can we provide anything useful in
        Ibis to help the user avoid them?

        :param left:
        :param right:
        """
        super().__init__(*self._maybe_cast_args(left, right))

    def _maybe_cast_args(self, left, right):
        # it might not be necessary?
        with suppress(com.IbisTypeError):
            return left, rlz.cast(right, left)

        with suppress(com.IbisTypeError):
            return rlz.cast(left, right), right

        return left, right

    def output_type(self):
        if not rlz.comparable(self.left, self.right):
            raise TypeError(
                'Arguments with datatype {} and {} are '
                'not comparable'.format(self.left.type(), self.right.type())
            )
        return rlz.shape_like(self.args, dt.boolean)


@public
class Equals(Comparison):
    pass


@public
class NotEquals(Comparison):
    pass


@public
class GreaterEqual(Comparison):
    pass


@public
class Greater(Comparison):
    pass


@public
class LessEqual(Comparison):
    pass


@public
class Less(Comparison):
    pass


@public
class IdenticalTo(Comparison):
    pass


@public
class Between(ValueOp, BooleanValueOp):
    arg = Arg(rlz.any)
    lower_bound = Arg(rlz.any)
    upper_bound = Arg(rlz.any)

    def output_type(self):
        arg, lower, upper = self.args

        if not (rlz.comparable(arg, lower) and rlz.comparable(arg, upper)):
            raise TypeError('Arguments are not comparable')

        return rlz.shape_like(self.args, dt.boolean)


@public
class Contains(ValueOp, BooleanValueOp):
    value = Arg(rlz.any)
    options = Arg(
        rlz.one_of(
            [
                rlz.value_list_of(rlz.any),
                rlz.set_,
                rlz.column(rlz.any),
                rlz.array_of(rlz.any),
            ]
        )
    )

    def output_type(self):
        return rlz.shape_like(list(self.flat_args()), dt.boolean)


@public
class NotContains(Contains):
    pass


@public
class Where(ValueOp):

    """
    Ternary case expression, equivalent to

    bool_expr.case()
             .when(True, true_expr)
             .else_(false_or_null_expr)
    """

    bool_expr = Arg(rlz.boolean)
    true_expr = Arg(rlz.any)
    false_null_expr = Arg(rlz.any)

    def output_type(self):
        return rlz.shape_like(self.bool_expr, self.true_expr.type())
