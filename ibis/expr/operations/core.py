from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional

from public import public
from typing_extensions import Any, Self, TypeVar

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis import util
from ibis.common.graph import Node as Traversable
from ibis.common.grounds import Concrete
from ibis.common.patterns import Coercible, CoercionError
from ibis.common.typing import DefaultTypeVars, VarTuple


@public
class Node(Concrete, Traversable):
    def equals(self, other) -> bool:
        if not isinstance(other, Node):
            raise TypeError(
                f"invalid equality comparison between Node and {type(other)}"
            )
        return self.__cached_equals__(other)

    @util.deprecated(as_of='4.0', instead='remove intermediate .op() calls')
    def op(self) -> Self:
        """Make `Node` backwards compatible with code that uses `Expr.op()`."""
        return self

    @abstractmethod
    def to_expr(self):
        ...

    # Avoid custom repr for performance reasons
    __repr__ = object.__repr__


@public
class Named(ABC):
    __slots__: VarTuple[str] = tuple()

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the operation.

        Returns
        -------
        str
        """


T = TypeVar("T", bound=dt.DataType, covariant=True)
S = TypeVar("S", bound=ds.DataShape, default=ds.Any, covariant=True)


@public
class Value(Node, Named, Coercible, DefaultTypeVars, Generic[T, S]):
    @classmethod
    def __coerce__(
        cls, value: Any, T: Optional[type] = None, S: Optional[type] = None
    ) -> Self:
        # note that S=Shape is unused here since the pattern will check the
        # shape of the value expression after executing Value.__coerce__()
        from ibis.expr.operations import Literal
        from ibis.expr.types import Expr

        if isinstance(value, Expr):
            value = value.op()
        if isinstance(value, Value):
            return value

        try:
            try:
                dtype = dt.dtype(T)
            except TypeError:
                dtype = dt.infer(value)
            return Literal(value, dtype=dtype)
        except TypeError:
            raise CoercionError(f"Unable to coerce {value!r} to Value[{T!r}]")

    # TODO(kszucs): cover it with tests
    # TODO(kszucs): figure out how to represent not named arguments
    @property
    def name(self) -> str:
        args = ", ".join(arg.name for arg in self.__args__ if isinstance(arg, Named))
        return f"{self.__class__.__name__}({args})"

    @property
    @abstractmethod
    def output_dtype(self) -> T:
        """Ibis datatype of the produced value expression.

        Returns
        -------
        dt.DataType
        """

    @property
    @abstractmethod
    def output_shape(self) -> S:
        """Shape of the produced value expression.

        Possible values are: "scalar" and "columnar"

        Returns
        -------
        rlz.Shape
        """

    def to_expr(self):
        import ibis.expr.types as ir

        if self.output_shape.is_columnar():
            typename = self.output_dtype.column
        else:
            typename = self.output_dtype.scalar

        return getattr(ir, typename)(self)


# convenience aliases
Scalar = Value[T, ds.Scalar]
Column = Value[T, ds.Columnar]


@public
class Alias(Value):
    arg: Value
    name: str

    output_shape = rlz.shape_like("arg")
    output_dtype = rlz.dtype_like("arg")


@public
class Unary(Value):
    """A unary operation."""

    arg: Value

    @property
    def output_shape(self) -> ds.DataShape:
        return self.arg.output_shape


@public
class Binary(Value):
    """A binary operation."""

    left: Value
    right: Value

    @property
    def output_shape(self) -> ds.DataShape:
        return max(self.left.output_shape, self.right.output_shape)


@public
class Argument(Value):
    name: str
    shape: ds.DataShape
    dtype: dt.DataType

    @property
    def output_dtype(self) -> dt.DataType:
        return self.dtype

    @property
    def output_shape(self) -> ds.DataShape:
        return self.shape


public(ValueOp=Value, UnaryOp=Unary, BinaryOp=Binary)
