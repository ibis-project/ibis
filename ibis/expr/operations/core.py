from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from public import public

import ibis.expr.rules as rlz
from ibis.common.grounds import Annotable, Comparable
from ibis.expr.rules import Shape
from ibis.util import UnnamedMarker


@public
class Node(Annotable, Comparable):
    @property
    def args(self):
        return self.__args__

    @property
    def argnames(self):
        return self.__argnames__

    def __equals__(self, other):
        return self.__args__ == other.__args__

    # TODO(kszucs): move to comparable
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


@public
class Named(ABC):

    __slots__ = tuple()

    @property
    @abstractmethod
    def name(self):
        """
        Name of the operation.

        Returns
        -------
        str
        """


@public
class Value(Node, Named):

    # TODO(kszucs): cover it with tests
    # TODO(kszucs): figure out how to represent not named arguments
    @property
    def name(self):
        args = ", ".join(
            arg.name for arg in self.__args__ if isinstance(arg, Named)
        )
        return f"{self.__class__.__name__}({args})"

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


@public
class NodeList(Node, Sequence[Node]):
    """
    Data structure for grouping arbitrary node objects.
    """

    values = rlz.tuple_of(rlz.instance_of(Node))

    @classmethod
    def __create__(self, *args):
        return super().__create__(values=args)

    @property
    def args(self):
        return self.values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __add__(self, other):
        values = self.values + tuple(other)
        return self.__class__(*values)

    def __radd__(self, other):
        values = tuple(other) + self.values
        return self.__class__(*values)

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.List(self)


public(ValueOp=Value, UnaryOp=Unary, BinaryOp=Binary, ValueList=NodeList)
