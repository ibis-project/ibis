from public import public

from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import BinaryOp, UnaryOp, ValueOp


@public
class LogicalBinaryOp(BinaryOp):
    left = rlz.boolean
    right = rlz.boolean

    output_dtype = dt.boolean


@public
class Not(UnaryOp):
    arg = rlz.boolean

    output_dtype = dt.boolean


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
class Comparison(BinaryOp):
    left = rlz.any
    right = rlz.any

    output_dtype = dt.boolean

    def __init__(self, left, right):
        """
        Casting rules for type promotions (for resolving the output type) may
        depend in some cases on the target backend.
        TODO: how will overflows be handled? Can we provide anything useful in
        Ibis to help the user avoid them?
        :param left:
        :param right:
        """
        if not rlz.comparable(left, right):
            raise TypeError(
                'Arguments with datatype {} and {} are '
                'not comparable'.format(left.type(), right.type())
            )
        super().__init__(left=left, right=right)


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
class Between(ValueOp):
    arg = rlz.any
    lower_bound = rlz.any
    upper_bound = rlz.any

    output_dtype = dt.boolean
    output_shape = rlz.shape_like("args")

    def __init__(self, arg, lower_bound, upper_bound):
        if not rlz.comparable(arg, lower_bound):
            raise TypeError(
                f'Argument with datatype {arg.type()} and lower bound '
                f'with datatype {lower_bound.type()} are not comparable'
            )
        if not rlz.comparable(arg, upper_bound):
            raise TypeError(
                f'Argument with datatype {arg.type()} and upper bound '
                f'with datatype {upper_bound.type()} are not comparable'
            )
        super().__init__(
            arg=arg, lower_bound=lower_bound, upper_bound=upper_bound
        )


@public
class Contains(ValueOp):
    value = rlz.any
    options = rlz.one_of(
        [
            rlz.value_list_of(rlz.any),
            rlz.set_,
            rlz.column(rlz.any),
            rlz.array_of(rlz.any),
        ]
    )

    output_dtype = dt.boolean
    output_shape = rlz.shape_like("args")


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

    bool_expr = rlz.boolean
    true_expr = rlz.any
    false_null_expr = rlz.any

    output_dtype = rlz.dtype_like("true_expr")
    output_shape = rlz.shape_like("bool_expr")
