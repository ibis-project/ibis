import collections
import functools
import itertools

import toolz
from public import public

from ... import util
from ...common import exceptions as com
from .. import rules as rlz
from ..signature import Annotable
from ..signature import Argument as Arg


def _safe_repr(x, memo=None):
    try:
        return x._repr(memo=memo)
    except AttributeError:
        return repr(x)


@public
def distinct_roots(*expressions):
    # TODO: move to analysis
    roots = toolz.concat(expr.op().root_tables() for expr in expressions)
    return list(toolz.unique(roots))


@public
def all_equal(left, right, cache=None):
    """Check whether two objects `left` and `right` are equal.

    Parameters
    ----------
    left : Union[object, Expr, Node]
    right : Union[object, Expr, Node]
    cache : Optional[Dict[Tuple[Node, Node], bool]]
        A dictionary indicating whether two Nodes are equal
    """
    if cache is None:
        cache = {}

    if util.is_iterable(left):
        # check that left and right are equal length iterables and that all
        # of their elements are equal
        return (
            util.is_iterable(right)
            and len(left) == len(right)
            and all(
                itertools.starmap(
                    functools.partial(all_equal, cache=cache), zip(left, right)
                )
            )
        )

    if hasattr(left, 'equals'):
        return left.equals(right, cache=cache)
    return left == right


def _maybe_get_op(value):
    try:
        return value.op()
    except AttributeError:
        return value


@public
class Node(Annotable):
    __slots__ = '_expr_cached', '_hash'

    def __repr__(self):
        return self._repr()

    def _repr(self, memo=None):
        if memo is None:
            from ibis.expr.format import FormatMemo

            memo = FormatMemo()

        opname = type(self).__name__
        pprint_args = []

        def _pp(x):
            return _safe_repr(x, memo=memo)

        for x in self.args:
            if isinstance(x, (tuple, list)):
                pp = repr(list(map(_pp, x)))
            else:
                pp = _pp(x)
            pprint_args.append(pp)

        return '{}({})'.format(opname, ', '.join(pprint_args))

    @property
    def inputs(self):
        return tuple(self.args)

    @property
    def exprs(self):
        from .. import types as ir

        return [arg for arg in self.args if isinstance(arg, ir.Expr)]

    def blocks(self):
        # The contents of this node at referentially distinct and may not be
        # analyzed deeper
        return False

    def flat_args(self):
        for arg in self.args:
            if not isinstance(arg, str) and isinstance(
                arg, collections.abc.Iterable
            ):
                yield from arg
            else:
                yield arg

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self._hash = hash(
                (type(self), *map(_maybe_get_op, self.flat_args()))
            )
        return self._hash

    def __eq__(self, other):
        return self.equals(other)

    def equals(self, other, cache=None):
        if cache is None:
            cache = {}

        key = self, other

        try:
            return cache[key]
        except KeyError:
            cache[key] = result = self is other or (
                type(self) == type(other)
                and all_equal(self.args, other.args, cache=cache)
            )
            return result

    def compatible_with(self, other):
        return self.equals(other)

    def is_ancestor(self, other):
        try:
            other = other.op()
        except AttributeError:
            pass

        return self.equals(other)

    def to_expr(self):
        if not hasattr(self, '_expr_cached'):
            self._expr_cached = self._make_expr()
        return self._expr_cached

    def _make_expr(self):
        klass = self.output_type()
        return klass(self)

    def output_type(self):
        """
        This function must resolve the output type of the expression and return
        the node wrapped in the appropriate ValueExpr type.
        """
        raise NotImplementedError


@public
class ValueOp(Node):
    def root_tables(self):
        return distinct_roots(*self.exprs)

    def resolve_name(self):
        raise com.ExpressionError(f'Expression is not named: {type(self)}')

    def has_resolved_name(self):
        return False


@public
class UnaryOp(ValueOp):
    """A unary operation."""

    arg = Arg(rlz.any)


@public
class BinaryOp(ValueOp):
    """A binary operation."""

    left = Arg(rlz.any)
    right = Arg(rlz.any)
