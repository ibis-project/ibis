from __future__ import annotations

import toolz
from public import public

from ...common import exceptions as com
from .. import rules as rlz
from ..signature import Annotable


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


def _compare_items(a, b):
    if hasattr(a, "equals"):
        return a.equals(b)
    elif isinstance(a, tuple):
        return _compare_tuples(a, b)
    else:
        return a == b


def _compare_tuples(a, b):
    if len(a) != len(b):
        return False
    return all(_compare_items(x, y) for x, y in zip(a, b))


@public
class Node(Annotable):
    __slots__ = ('_expr_cached',)

    def __repr__(self):
        return self._repr()

    def __equals__(self, other):
        return (
            type(self) == type(other)
            and self._hash == other._hash
            and _compare_tuples(self.args, other.args)
        )

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

    def compatible_with(self, other):
        return self.equals(other)

    def is_ancestor(self, other):
        try:
            other = other.op()
        except AttributeError:
            pass

        return self.equals(other)

    def to_expr(self):
        try:
            result = self._expr_cached
        except AttributeError:
            result = self._make_expr()
            object.__setattr__(self, "_expr_cached", result)
        return result

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

    arg = rlz.any


@public
class BinaryOp(ValueOp):
    """A binary operation."""

    left = rlz.any
    right = rlz.any
