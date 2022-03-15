from __future__ import annotations

from typing import Any

import toolz
from public import public

from ...common import exceptions as com
from .. import rules as rlz
from ..signature import Annotable


@public
def distinct_roots(*expressions):
    # TODO: move to analysis
    roots = toolz.concat(expr.op().root_tables() for expr in expressions)
    return list(toolz.unique(roots))


@public
class Node(Annotable):
    __slots__ = ("_flat_ops",)

    def __post_init__(self):
        import ibis.expr.types as ir

        super().__post_init__()
        object.__setattr__(
            self,
            "_flat_ops",
            tuple(
                arg.op()
                for arg in self.flat_args()
                if isinstance(arg, ir.Expr)
            ),
        )

    def _type_check(self, other: Any) -> None:
        if not isinstance(other, Node):
            raise TypeError(
                f"Cannot compare non-Node object {type(other)} with Node"
            )

    @property
    def inputs(self):
        return self.args

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
        return self._make_expr()

    def _make_expr(self):
        klass = self.output_type()
        return klass(self)

    def output_type(self):
        """Resolve the output type of the expression."""
        raise NotImplementedError(
            f"output_type not implemented for {type(self)}"
        )


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
