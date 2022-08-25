from __future__ import annotations

from abc import abstractmethod

from public import public

import ibis.expr.rules as rlz
from ibis.common.exceptions import ExpressionError
from ibis.common.grounds import Annotable, Comparable
from ibis.expr.rules import Shape
from ibis.expr.schema import Schema
from ibis.util import UnnamedMarker, is_iterable


@public
class Node(Annotable, Comparable):
    def __equals__(self, other):
        return self.args == other.args

    def equals(self, other):
        if not isinstance(other, Node):
            raise TypeError(
                "invalid equality comparison between Node and "
                f"{type(other)}"
            )
        return self.__cached_equals__(other)

    @abstractmethod
    def to_expr(self):
        ...

    # TODO(kszucs): introduce a HasName schema, or NamedValue with a .name
    # abstractproperty
    def resolve_name(self):
        raise ExpressionError(f'Expression is not named: {type(self)}')

    def has_resolved_name(self):
        return False

    # TODO(kszucs): remove this method entirely
    def flat_args(self):
        for arg in self.args:
            if not isinstance(arg, Schema) and is_iterable(arg):
                yield from arg
            else:
                yield arg


@public
class Value(Node):
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

    def to_expr(self):
        if self.output_shape is Shape.COLUMNAR:
            return self.output_dtype.column(self)
        else:
            return self.output_dtype.scalar(self)


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
        return self.arg.output_shape


@public
class Binary(Value):
    """A binary operation."""

    left = rlz.any
    right = rlz.any

    @property
    def output_shape(self):
        return max(self.left.output_shape, self.right.output_shape)


public(ValueOp=Value, UnaryOp=Unary, BinaryOp=Binary)
