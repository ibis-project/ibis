from __future__ import annotations

from abc import abstractmethod

import toolz
from public import public

from ibis import util
from ibis.common.exceptions import ExpressionError
from ibis.common.grounds import Comparable
from ibis.common.validators import immutable_property
from ibis.expr import rules as rlz
from ibis.expr import types as ir
from ibis.expr.rules import Shape
from ibis.expr.schema import Schema
from ibis.expr.signature import Annotable
from ibis.util import UnnamedMarker, is_iterable


@public
def distinct_roots(*expressions):
    # TODO: move to analysis
    roots = toolz.concat(expr.op().root_tables() for expr in expressions)
    return list(toolz.unique(roots))


def _compare_items(a, b):
    try:
        return a.equals(b)
    except AttributeError:
        if isinstance(a, tuple):
            return _compare_tuples(a, b)
        else:
            return a == b


def _compare_tuples(a, b):
    if len(a) != len(b):
        return False
    return all(map(_compare_items, a, b))


@public
class Node(Annotable, Comparable):
    @immutable_property
    def _flat_ops(self):
        import ibis.expr.types as ir

        return tuple(
            arg.op() for arg in self.flat_args() if isinstance(arg, ir.Expr)
        )

    def __equals__(self, other):
        return self._hash == other._hash and _compare_tuples(
            self.args, other.args
        )

    def equals(self, other):
        if not isinstance(other, Node):
            raise TypeError(
                "invalid equality comparison between Node and "
                f"{type(other)}"
            )
        return self.__cached_equals__(other)

    @property
    def inputs(self):
        return self.args

    @property
    def exprs(self):
        return [arg for arg in self.args if isinstance(arg, ir.Expr)]

    def blocks(self):
        # The contents of this node at referentially distinct and may not be
        # analyzed deeper
        return False

    @util.deprecated(instead="", version="4.0.0")
    def compatible_with(self, other):  # pragma: no cover
        return self.equals(other)

    @util.deprecated(version="4.0.0", instead="")
    def is_ancestor(self, other):
        try:
            other = other.op()
        except AttributeError:
            pass

        return self.equals(other)

    def to_expr(self):
        return self.output_type(self)

    def resolve_name(self):
        raise ExpressionError(f'Expression is not named: {type(self)}')

    def has_resolved_name(self):
        return False

    @property
    def output_type(self):
        """Resolve the output type of the expression."""
        raise NotImplementedError(
            f"output_type not implemented for {type(self)}"
        )

    def flat_args(self):
        for arg in self.args:
            if not isinstance(arg, Schema) and is_iterable(arg):
                yield from arg
            else:
                yield arg


@public
class Value(Node):
    def root_tables(self):
        return distinct_roots(*self.exprs)

    @property
    @abstractmethod
    def output_dtype(self):
        """
        Ibis datatype of the produced value expression.

        Returns
        -------
        dt.DataType
        """

    @property
    @abstractmethod
    def output_shape(self):
        """
        Shape of the produced value expression.

        Possible values are: "scalar" and "columnar"

        Returns
        -------
        rlz.Shape
        """

    @property
    def output_type(self):
        if self.output_shape is Shape.COLUMNAR:
            return self.output_dtype.column
        else:
            return self.output_dtype.scalar


@public
class Alias(Value):
    arg = rlz.any
    name = rlz.instance_of((str, UnnamedMarker))

    output_shape = rlz.shape_like("arg")
    output_dtype = rlz.dtype_like("arg")

    def has_resolved_name(self):
        return True

    def resolve_name(self):
        return self.name


@public
class Unary(Value):
    """A unary operation."""

    arg = rlz.any

    @property
    def output_shape(self):
        return self.arg.op().output_shape


@public
class Binary(Value):
    """A binary operation."""

    left = rlz.any
    right = rlz.any

    @property
    def output_shape(self):
        return max(self.left.op().output_shape, self.right.op().output_shape)


public(ValueOp=Value, UnaryOp=Unary, BinaryOp=Binary)
