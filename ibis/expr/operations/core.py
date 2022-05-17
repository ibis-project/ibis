from __future__ import annotations

from abc import abstractmethod

import toolz
from matchpy import Arity, Operation, Wildcard
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


def _erase_exprs(arg):
    """
    Remove intermediate expressions.
    """
    if isinstance(arg, ir.Expr):
        return arg.op()
    elif isinstance(arg, tuple):
        return tuple(map(_erase_exprs, arg))
    else:
        return arg


def _create_exprs(arg):
    if isinstance(arg, Node):
        return arg.to_expr()
    elif isinstance(arg, tuple):
        return tuple(map(_create_exprs, arg))
    else:
        return arg


# TODO(kszucs): should rename to Operator
@public
class Node(Annotable, Comparable):

    __slots__ = ('args',)

    ####################### MATCHPY API ###############################  # noqa

    def __init_subclass__(
        cls,
        /,
        name=None,
        arity=False,
        associative=False,
        commutative=False,
        one_identity=False,
        infix=False,
        **kwargs,
    ):
        # TODO(kszucs): raise if class already has these attributes
        # cls.name = name or cls.__name__
        cls.arity = arity or Arity(len(cls.__argnames__), True)
        cls.associative = associative
        cls.commutative = commutative
        cls.one_identity = one_identity
        cls.infix = infix

    # TODO(kszucs): restore iter and len
    def __iter__(self):
        # returns with an iterator of Nodes
        return iter(self.__args__)

    def __len__(self):
        return len(self.__args__)

    # def __getitem__(self, key):
    #     index = self.__argnames__.index(key)
    #     return self.__args__[index]

    # def __contains__(self, key):
    #     return key in self.__argnames__

    # @property
    # def operands(self):
    #     raise NotImplementedError()

    # def collect_variables(self):
    #     raise NotImplementedError()

    # def collect_symbols(self, symbols):
    #     raise NotImplementedError()

    @classmethod
    def __create__(cls, *args, bypass_validation=False, **kwargs):
        bound = cls.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()

        # bound the signature to the passed arguments and apply the validators
        # before passing the arguments, so self.__init__() receives already
        # validated arguments as keywords
        kwargs = {}
        for name, value in bound.arguments.items():
            param = cls.__signature__.parameters[name]
            # TODO(kszucs): provide more error context on failure
            # FIXME(kszucs): hack for supporting matchpy wildcards for pattern
            # matching
            # print(name, type(value))
            # HACK
            if bypass_validation or isinstance(value, (Wildcard, Node)):
                kwargs[name] = value
            else:
                kwargs[name] = param.validate(kwargs, value)

        # construct the instance by passing the validated keyword arguments
        return super(Annotable, cls).__create__(**kwargs)

    ####################### MATCHPY API END ###########################  # noqa

    def __init__(self, **kwargs):
        kwargs = {k: _erase_exprs(v) for k, v in kwargs.items()}
        object.__setattr__(
            self, 'args', tuple(map(_create_exprs, kwargs.values()))
        )
        super().__init__(**kwargs)

    def __post_init__(self):
        for arg in self.__args__:
            assert not isinstance(arg, ir.Expr)
        for arg in self.args:
            assert not isinstance(arg, Node)

    @property
    def argnames(self):
        return self.__argnames__

    def get(self, name):
        return super().__getattr__(name)

    def __getattribute__(self, name):
        arg = super().__getattribute__(name)
        if name in type(self).__argnames__:
            arg = _create_exprs(arg)
        return arg

    def __reduce__(self):
        kwargs = dict(zip(self.argnames, self.args))
        return (self._reconstruct, (kwargs,))

    @immutable_property
    def _flat_ops(self):
        import ibis.expr.types as ir

        return tuple(
            arg.op() for arg in self.flat_args() if isinstance(arg, ir.Expr)
        )

    def __equals__(self, other):
        return self._hash == other._hash and self.__args__ == other.__args__

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


Operation.register(Node)


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



@public
class List(Node):
    """Data structure for a list of expressions"""

    values = rlz.tuple_of(rlz.instance_of(ir.Expr))

    output_type = ir.List

    def root_tables(self):
        return distinct_roots(*self.values)


@public
class ValueList(List, Value):
    """Data structure for a list of value expressions"""

    values = rlz.tuple_of(rlz.any)

    output_type = ir.ValueList
    output_dtype = rlz.dtype_like("values")
    output_shape = rlz.shape_like("values")

    # def root_tables(self):
    #     return distinct_roots(*self.values)


public(UnaryOp=Unary, BinaryOp=Binary, ValueOp=Value)
