from __future__ import annotations

import copy
import pickle
import sys
import weakref
from abc import ABCMeta
from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeVar, Union

import pytest

from ibis.common.annotations import (
    Argument,
    Parameter,
    Signature,
    ValidationError,
    argument,
    attribute,
    optional,
    varargs,
    varkwargs,
)
from ibis.common.collections import Mapping, Sequence
from ibis.common.grounds import (
    Abstract,
    Annotable,
    AnnotableMeta,
    Comparable,
    Concrete,
    Immutable,
    Singleton,
)
from ibis.common.patterns import (
    Any,
    As,
    CoercedTo,
    Coercible,
    InstanceOf,
    Option,
    Pattern,
    TupleOf,
)
from ibis.tests.util import assert_pickle_roundtrip

if TYPE_CHECKING:
    from typing_extensions import Self

is_any = InstanceOf(object)
is_bool = InstanceOf(bool)
is_float = InstanceOf(float)
is_int = InstanceOf(int)
is_str = InstanceOf(str)
is_list = InstanceOf(list)


class Op(Annotable):
    pass


class Value(Op):
    arg = InstanceOf(object)


class StringOp(Value):
    arg = InstanceOf(str)


class BetweenSimple(Annotable):
    value = is_int
    lower = optional(is_int, default=0)
    upper = optional(is_int, default=None)


class BetweenWithExtra(Annotable):
    extra = attribute(is_int)
    value = is_int
    lower = optional(is_int, default=0)
    upper = optional(is_int, default=None)


class BetweenWithCalculated(Concrete):
    value = is_int
    lower = optional(is_int, default=0)
    upper = optional(is_int, default=None)

    @attribute
    def calculated(self):
        return self.value + self.lower


class VariadicArgs(Concrete):
    args = varargs(is_int)


class VariadicKeywords(Concrete):
    kwargs = varkwargs(is_int)


class VariadicArgsAndKeywords(Concrete):
    args = varargs(is_int)
    kwargs = varkwargs(is_int)


T = TypeVar("T", covariant=True)
K = TypeVar("K", covariant=True)
V = TypeVar("V", covariant=True)


class List(Concrete, Sequence[T], Coercible):
    @classmethod
    def __coerce__(self, values, T=None):
        values = tuple(values)
        if values:
            head, *rest = values
            return ConsList(head, rest)
        else:
            return EmptyList()


class EmptyList(List[T]):
    def __getitem__(self, key):
        raise IndexError(key)

    def __len__(self):
        return 0


class ConsList(List[T]):
    head: T
    rest: List[T]

    def __getitem__(self, key):
        if key == 0:
            return self.head
        else:
            return self.rest[key - 1]

    def __len__(self):
        return len(self.rest) + 1


class Map(Concrete, Mapping[K, V], Coercible):
    @classmethod
    def __coerce__(self, pairs, K=None, V=None):
        pairs = dict(pairs)
        if pairs:
            head_key = next(iter(pairs))
            head_value = pairs.pop(head_key)
            rest = pairs
            return ConsMap((head_key, head_value), rest)
        else:
            return EmptyMap()


class EmptyMap(Map[K, V]):
    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class ConsMap(Map[K, V]):
    head: tuple[K, V]
    rest: Map[K, V]

    def __getitem__(self, key):
        if key == self.head[0]:
            return self.head[1]
        else:
            return self.rest[key]

    def __iter__(self):
        yield self.head[0]
        yield from self.rest

    def __len__(self):
        return len(self.rest) + 1


class Integer(int, Coercible):
    @classmethod
    def __coerce__(cls, value):
        return Integer(value)


class Float(float, Coercible):
    @classmethod
    def __coerce__(cls, value):
        return Float(value)


class MyExpr(Concrete):
    a: Integer
    b: List[Float]
    c: Map[str, Integer]


class MyInt(int, Coercible):
    @classmethod
    def __coerce__(cls, value):
        return cls(value)


class MyFloat(float, Coercible):
    @classmethod
    def __coerce__(cls, value):
        return cls(value)


J = TypeVar("J", bound=MyInt, covariant=True)
F = TypeVar("F", bound=MyFloat, covariant=True)
N = TypeVar("N", bound=Union[MyInt, MyFloat], covariant=True)


class MyValue(Annotable, Generic[J, F]):
    integer: J
    floating: F
    numeric: N


def test_annotable():
    class Between(BetweenSimple):
        pass

    assert not issubclass(type(Between), ABCMeta)
    assert type(Between) is AnnotableMeta

    argnames = ("value", "lower", "upper")
    signature = BetweenSimple.__signature__
    assert isinstance(signature, Signature)
    assert tuple(signature.parameters.keys()) == argnames
    assert BetweenSimple.__slots__ == argnames

    obj = BetweenSimple(10, lower=2)
    assert obj.value == 10
    assert obj.lower == 2
    assert obj.upper is None
    assert obj.__argnames__ == argnames
    assert obj.__slots__ == ("value", "lower", "upper")
    assert not hasattr(obj, "__dict__")
    assert obj.__module__ == __name__
    assert type(obj).__qualname__ == "BetweenSimple"

    # test that a child without additional arguments doesn't have __dict__
    obj = Between(10, lower=2)
    assert obj.__slots__ == tuple()
    assert not hasattr(obj, "__dict__")
    assert obj.__module__ == __name__
    assert type(obj).__qualname__ == "test_annotable.<locals>.Between"

    copied = copy.copy(obj)
    assert obj == copied
    assert obj is not copied

    copied = obj.copy()
    assert obj == copied
    assert obj is not copied

    obj2 = Between(10, lower=8)
    assert obj.copy(lower=8) == obj2


def test_annotable_with_bound_typevars_properly_coerce_values():
    v = MyValue(1.1, 2.2, 3.3)
    assert isinstance(v.integer, MyInt)
    assert v.integer == 1
    assert isinstance(v.floating, MyFloat)
    assert v.floating == 2.2
    assert isinstance(v.numeric, MyInt)
    assert v.numeric == 3


def test_annotable_with_additional_attributes():
    a = BetweenWithExtra(10, lower=2)
    b = BetweenWithExtra(10, lower=2)
    assert a == b
    assert a is not b

    a.extra = 1
    assert a.extra == 1
    assert a != b

    assert a == pickle.loads(pickle.dumps(a))


def test_annotable_is_mutable_by_default():
    # TODO(kszucs): more exhaustive testing of mutability, e.g. setting
    # optional value to None doesn't set to the default value
    class Op(Annotable):
        __slots__ = ("custom",)

        a = is_int
        b = is_int

    p = Op(1, 2)
    assert p.a == 1
    p.a = 3
    assert p.a == 3
    assert p == Op(a=3, b=2)

    # test that non-annotable attributes can be set as well
    p.custom = 1
    assert p.custom == 1


def test_annotable_with_type_annotations() -> None:
    # TODO(kszucs): bar: str = None  # should raise
    class Op1(Annotable):
        foo: int
        bar: str = ""

    p = Op1(1)
    assert p.foo == 1
    assert not p.bar

    class Op2(Annotable):
        bar: str = None

    with pytest.raises(ValidationError):
        Op2()


class RecursiveNode(Annotable):
    child: Optional[Self] = None


def test_annotable_with_self_typehint() -> None:
    node = RecursiveNode(RecursiveNode(RecursiveNode(None)))
    assert isinstance(node, RecursiveNode)
    assert isinstance(node.child, RecursiveNode)
    assert isinstance(node.child.child, RecursiveNode)
    assert node.child.child.child is None

    with pytest.raises(ValidationError):
        RecursiveNode(1)


def test_annotable_with_recursive_generic_type_annotations():
    # testing cons list
    pattern = Pattern.from_typehint(List[Integer])
    values = ["1", 2.0, 3]
    result = pattern.match(values, {})
    expected = ConsList(1, ConsList(2, ConsList(3, EmptyList())))
    assert result == expected
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert len(result) == 3
    with pytest.raises(IndexError):
        result[3]

    # testing cons map
    pattern = Pattern.from_typehint(Map[Integer, Float])
    values = {"1": 2, 3: "4.0", 5: 6.0}
    result = pattern.match(values, {})
    expected = ConsMap((1, 2.0), ConsMap((3, 4.0), ConsMap((5, 6.0), EmptyMap())))
    assert result == expected
    assert result[1] == 2.0
    assert result[3] == 4.0
    assert result[5] == 6.0
    assert len(result) == 3
    with pytest.raises(KeyError):
        result[7]

    # testing both encapsulated in a class
    expr = MyExpr(a="1", b=["2.0", 3, True], c={"a": "1", "b": 2, "c": 3.0})
    assert expr.a == 1
    assert expr.b == ConsList(2.0, ConsList(3.0, ConsList(1.0, EmptyList())))
    assert expr.c == ConsMap(("a", 1), ConsMap(("b", 2), ConsMap(("c", 3), EmptyMap())))


def test_composition_of_annotable_and_immutable():
    class AnnImm(Annotable, Immutable):
        value = is_int
        lower = optional(is_int, default=0)
        upper = optional(is_int, default=None)

    class ImmAnn(Immutable, Annotable):
        # this is the preferable method resolution order
        value = is_int
        lower = optional(is_int, default=0)
        upper = optional(is_int, default=None)

    obj = AnnImm(3, lower=0, upper=4)
    with pytest.raises(AttributeError):
        obj.value = 1

    obj = ImmAnn(3, lower=0, upper=4)
    with pytest.raises(AttributeError):
        obj.value = 1


def test_composition_of_annotable_and_comparable():
    class Between(Comparable, Annotable):
        value = is_int
        lower = optional(is_int, default=0)
        upper = optional(is_int, default=None)

        def __equals__(self, other):
            return (
                self.value == other.value
                and self.lower == other.lower
                and self.upper == other.upper
            )

    a = Between(3, lower=0, upper=4)
    b = Between(3, lower=0, upper=4)
    c = Between(2, lower=0, upper=4)

    assert Between.__eq__ is Comparable.__eq__
    assert a == b
    assert b == a
    assert a != c
    assert c != a
    assert a.__equals__(b)
    assert a.__cached_equals__(b)
    assert not a.__equals__(c)
    assert not a.__cached_equals__(c)


def test_maintain_definition_order():
    class Between(Annotable):
        value = is_int
        lower = optional(is_int, default=0)
        upper = optional(is_int, default=None)

    param_names = list(Between.__signature__.parameters.keys())
    assert param_names == ["value", "lower", "upper"]


def test_signature_inheritance():
    class IntBinop(Annotable):
        left = is_int
        right = is_int

    class FloatAddRhs(IntBinop):
        right = is_float

    class FloatAddClip(FloatAddRhs):
        left = is_float
        clip_lower = optional(is_int, default=0)
        clip_upper = optional(is_int, default=10)

    class IntAddClip(FloatAddClip, IntBinop):
        pass

    assert IntBinop.__signature__ == Signature(
        [
            Parameter("left", annotation=Argument(is_int)),
            Parameter("right", annotation=Argument(is_int)),
        ]
    )

    assert FloatAddRhs.__signature__ == Signature(
        [
            Parameter("left", annotation=Argument(is_int)),
            Parameter("right", annotation=Argument(is_float)),
        ]
    )

    assert FloatAddClip.__signature__ == Signature(
        [
            Parameter("left", annotation=Argument(is_float)),
            Parameter("right", annotation=Argument(is_float)),
            Parameter("clip_lower", annotation=optional(is_int, default=0)),
            Parameter("clip_upper", annotation=optional(is_int, default=10)),
        ]
    )

    assert IntAddClip.__signature__ == Signature(
        [
            Parameter("left", annotation=Argument(is_int)),
            Parameter("right", annotation=Argument(is_int)),
            Parameter("clip_lower", annotation=optional(is_int, default=0)),
            Parameter("clip_upper", annotation=optional(is_int, default=10)),
        ]
    )


def test_positional_argument_reordering():
    class Farm(Annotable):
        ducks = is_int
        donkeys = is_int
        horses = is_int
        goats = is_int
        chickens = is_int

    class NoHooves(Farm):
        horses = optional(is_int, default=0)
        goats = optional(is_int, default=0)
        donkeys = optional(is_int, default=0)

    f1 = Farm(1, 2, 3, 4, 5)
    f2 = Farm(1, 2, goats=4, chickens=5, horses=3)
    f3 = Farm(1, 0, 0, 0, 100)
    assert f1 == f2
    assert f1 != f3

    g1 = NoHooves(1, 2, donkeys=-1)
    assert g1.ducks == 1
    assert g1.chickens == 2
    assert g1.donkeys == -1
    assert g1.horses == 0
    assert g1.goats == 0


def test_keyword_argument_reordering():
    class Alpha(Annotable):
        a = is_int
        b = is_int

    class Beta(Alpha):
        c = is_int
        d = optional(is_int, default=0)
        e = is_int

    obj = Beta(1, 2, 3, 4)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.e == 4
    assert obj.d == 0

    obj = Beta(1, 2, 3, 4, 5)
    assert obj.d == 5
    assert obj.e == 4


def test_variadic_argument_reordering():
    class Test(Annotable):
        a = is_int
        b = is_int
        args = varargs(is_int)

    class Test2(Test):
        c = is_int
        args = varargs(is_int)

    with pytest.raises(ValidationError, match="missing a required argument: 'c'"):
        Test2(1, 2)

    a = Test2(1, 2, 3)
    assert a.a == 1
    assert a.b == 2
    assert a.c == 3
    assert a.args == ()

    b = Test2(*range(5))
    assert b.a == 0
    assert b.b == 1
    assert b.c == 2
    assert b.args == (3, 4)

    msg = "only one variadic \\*args parameter is allowed"
    with pytest.raises(TypeError, match=msg):

        class Test3(Test):
            another_args = varargs(is_int)


def test_variadic_keyword_argument_reordering():
    class Test(Annotable):
        a = is_int
        b = is_int
        options = varkwargs(is_int)

    class Test2(Test):
        c = is_int
        options = varkwargs(is_int)

    with pytest.raises(ValidationError, match="missing a required argument: 'c'"):
        Test2(1, 2)

    a = Test2(1, 2, c=3)
    assert a.a == 1
    assert a.b == 2
    assert a.c == 3
    assert a.options == {}

    b = Test2(1, 2, c=3, d=4, e=5)
    assert b.a == 1
    assert b.b == 2
    assert b.c == 3
    assert b.options == {"d": 4, "e": 5}

    msg = "only one variadic \\*\\*kwargs parameter is allowed"
    with pytest.raises(TypeError, match=msg):

        class Test3(Test):
            another_options = varkwargs(is_int)


def test_variadic_argument():
    class Test(Annotable):
        a = is_int
        b = is_int
        args = varargs(is_int)

    assert Test(1, 2).args == ()
    assert Test(1, 2, 3).args == (3,)
    assert Test(1, 2, 3, 4, 5).args == (3, 4, 5)


def test_variadic_keyword_argument():
    class Test(Annotable):
        first = is_int
        second = is_int
        options = varkwargs(is_int)

    assert Test(1, 2).options == {}
    assert Test(1, 2, a=3).options == {"a": 3}
    assert Test(1, 2, a=3, b=4, c=5).options == {"a": 3, "b": 4, "c": 5}


def test_concrete_copy_with_variadic_argument():
    class Test(Annotable):
        a = is_int
        b = is_int
        args = varargs(is_int)

    t = Test(1, 2, 3, 4, 5)
    assert t.a == 1
    assert t.b == 2
    assert t.args == (3, 4, 5)

    u = t.copy(a=6, args=(8, 9, 10))
    assert u.a == 6
    assert u.b == 2
    assert u.args == (8, 9, 10)


def test_concrete_pickling_variadic_arguments():
    v = VariadicArgs(1, 2, 3, 4, 5)
    assert v.args == (1, 2, 3, 4, 5)
    assert_pickle_roundtrip(v)

    v = VariadicKeywords(a=3, b=4, c=5)
    assert v.kwargs == {"a": 3, "b": 4, "c": 5}
    assert_pickle_roundtrip(v)

    v = VariadicArgsAndKeywords(1, 2, 3, 4, 5, a=3, b=4, c=5)
    assert v.args == (1, 2, 3, 4, 5)
    assert v.kwargs == {"a": 3, "b": 4, "c": 5}
    assert_pickle_roundtrip(v)


def test_dont_copy_default_argument():
    default = tuple()

    class Op(Annotable):
        arg = optional(InstanceOf(tuple), default=default)

    op = Op()
    assert op.arg is default


def test_copy_mutable_with_default_attribute():
    class Test(Annotable):
        a = attribute(InstanceOf(dict), default={})
        b = argument(InstanceOf(str))  # required argument

        @attribute
        def c(self):
            return self.b.upper()

    t = Test("t")
    assert t.a == {}
    assert t.b == "t"
    assert t.c == "T"

    with pytest.raises(ValidationError):
        t.a = 1
    t.a = {"map": "ping"}
    assert t.a == {"map": "ping"}

    assert t.copy() == t

    u = t.copy(b="u")
    assert u.b == "u"
    assert u.c == "T"
    assert u.a == {"map": "ping"}

    x = t.copy(a={"emp": "ty"})
    assert x.a == {"emp": "ty"}
    assert x.b == "t"


def test_slots_are_inherited_and_overridable():
    class Op(Annotable):
        __slots__ = ("_cache",)  # first definition
        arg = Any()

    class StringOp(Op):
        arg = CoercedTo(str)  # new overridden slot

    class StringSplit(StringOp):
        sep = CoercedTo(str)  # new slot

    class StringJoin(StringOp):
        __slots__ = ("_memoize",)  # new slot
        sep = CoercedTo(str)  # new overridden slot

    assert Op.__slots__ == ("_cache", "arg")
    assert StringOp.__slots__ == ("arg",)
    assert StringSplit.__slots__ == ("sep",)
    assert StringJoin.__slots__ == ("_memoize", "sep")


def test_multiple_inheritance():
    # multiple inheritance is allowed only if one of the parents has non-empty
    # __slots__ definition, otherwise python will raise lay-out conflict

    class Op(Annotable):
        __slots__ = ("_hash",)

    class Value(Annotable):
        arg = InstanceOf(object)

    class Reduction(Value):
        pass

    class UDF(Value):
        func = InstanceOf(Callable)

    class UDAF(UDF, Reduction):
        arity = is_int

    class A(Annotable):
        a = is_int

    class B(Annotable):
        b = is_int

    msg = "multiple bases have instance lay-out conflict"
    with pytest.raises(TypeError, match=msg):

        class AB(A, B):
            ab = is_int

    assert UDAF.__slots__ == ("arity",)
    strlen = UDAF(arg=2, func=lambda value: len(str(value)), arity=1)
    assert strlen.arg == 2
    assert strlen.arity == 1


@pytest.mark.parametrize(
    "obj",
    [
        StringOp("something"),
        StringOp(arg="something"),
    ],
)
def test_pickling_support(obj):
    assert_pickle_roundtrip(obj)


def test_multiple_inheritance_argument_order():
    class Value(Annotable):
        arg = is_any

    class VersionedOp(Value):
        version = is_int

    class Reduction(Annotable):
        pass

    class Sum(VersionedOp, Reduction):
        where = optional(is_bool, default=False)

    assert tuple(Sum.__signature__.parameters.keys()) == ("arg", "version", "where")


def test_multiple_inheritance_optional_argument_order():
    class Value(Annotable):
        pass

    class ConditionalOp(Annotable):
        where = optional(is_bool, default=False)

    class Between(Value, ConditionalOp):
        min = is_int
        max = is_int
        how = optional(is_str, default="strict")

    assert tuple(Between.__signature__.parameters.keys()) == (
        "min",
        "max",
        "how",
        "where",
    )


def test_immutability():
    class Value(Annotable, Immutable):
        a = is_int

    op = Value(1)
    with pytest.raises(AttributeError):
        op.a = 3


class BaseValue(Annotable):
    i = is_int
    j = attribute(is_int)


class Value2(BaseValue):
    @attribute
    def k(self):
        return 3


class Value3(BaseValue):
    k = attribute(is_int, default=3)


class Value4(BaseValue):
    k = attribute(Option(is_int), default=None)


def test_annotable_with_dict_slot():
    class Flexible(Annotable):
        __slots__ = ("__dict__",)

    v = Flexible()
    v.a = 1
    v.b = 2
    assert v.a == 1
    assert v.b == 2


def test_annotable_attribute():
    with pytest.raises(ValidationError, match="too many positional arguments"):
        BaseValue(1, 2)

    v = BaseValue(1)
    assert v.__slots__ == ("i", "j")
    assert v.i == 1
    assert not hasattr(v, "j")
    v.j = 2
    assert v.j == 2

    with pytest.raises(ValidationError):
        v.j = "foo"


def test_annotable_attribute_init():
    assert Value2.__slots__ == ("k",)
    v = Value2(1)

    assert v.i == 1
    assert not hasattr(v, "j")
    v.j = 2
    assert v.j == 2
    assert v.k == 3

    v = Value3(1)
    assert v.k == 3

    v = Value4(1)
    assert v.k is None


def test_annotable_mutability_and_serialization():
    v_ = BaseValue(1)
    v_.j = 2
    v = BaseValue(1)
    v.j = 2
    assert v_ == v
    assert v_.j == v.j == 2

    assert repr(v) == "BaseValue(i=1)"
    w = pickle.loads(pickle.dumps(v))
    assert w.i == 1
    assert w.j == 2
    assert v == w

    v.j = 4
    assert v_ != v
    w = pickle.loads(pickle.dumps(v))
    assert w == v
    assert repr(w) == "BaseValue(i=1)"


def test_initialized_attribute_basics():
    class Value(Annotable):
        a = is_int

        @attribute
        def double_a(self):
            return 2 * self.a

    op = Value(1)
    assert op.a == 1
    assert op.double_a == 2
    assert len(Value.__attributes__) == 1
    assert "double_a" in Value.__slots__


def test_initialized_attribute_with_validation():
    class Value(Annotable):
        a = is_int

        @attribute(int)
        def double_a(self):
            return 2 * self.a

    op = Value(1)
    assert op.a == 1
    assert op.double_a == 2
    assert len(Value.__attributes__) == 1
    assert "double_a" in Value.__slots__

    op.double_a = 3
    assert op.double_a == 3

    with pytest.raises(ValidationError):
        op.double_a = "foo"


def test_initialized_attribute_mixed_with_classvar():
    class Value(Annotable):
        arg = is_int

        shape = "like-arg"
        dtype = "like-arg"

    class Reduction(Value):
        shape = "scalar"

    class Variadic(Value):
        @attribute
        def shape(self):
            if self.arg > 10:
                return "columnar"
            else:
                return "scalar"

    r = Reduction(1)
    assert r.shape == "scalar"
    assert "shape" not in r.__slots__

    v = Variadic(1)
    assert v.shape == "scalar"
    assert "shape" in v.__slots__

    v = Variadic(100)
    assert v.shape == "columnar"
    assert "shape" in v.__slots__


def test_composition_of_annotable_and_singleton() -> None:
    class AnnSing(Annotable, Singleton):
        value = CoercedTo(int)

    class SingAnn(Singleton, Annotable):
        # this is the preferable method resolution order
        value = CoercedTo(int)

    # arguments looked up after validation
    obj1 = AnnSing("3")
    assert AnnSing("3") is obj1
    assert AnnSing(3) is obj1
    assert AnnSing(3.0) is obj1

    # arguments looked up before validation
    obj2 = SingAnn("3")
    assert SingAnn("3") is obj2
    obj3 = SingAnn(3)
    assert obj3 is not obj2
    assert SingAnn(3) is obj3


def test_concrete():
    assert BetweenWithCalculated.__mro__ == (
        BetweenWithCalculated,
        Concrete,
        Immutable,
        Comparable,
        Annotable,
        Abstract,
        object,
    )

    assert BetweenWithCalculated.__create__.__func__ is Annotable.__create__.__func__
    assert BetweenWithCalculated.__eq__ is Comparable.__eq__
    assert BetweenWithCalculated.__argnames__ == ("value", "lower", "upper")

    # annotable
    obj = BetweenWithCalculated(10, lower=5, upper=15)
    obj2 = BetweenWithCalculated(10, lower=5, upper=15)
    assert obj.value == 10
    assert obj.lower == 5
    assert obj.upper == 15
    assert obj.calculated == 15
    assert obj == obj2
    assert obj is not obj2
    assert obj != (10, 5, 15)
    assert obj.__args__ == (10, 5, 15)
    assert obj.args == (10, 5, 15)
    assert obj.argnames == ("value", "lower", "upper")

    # immutable
    with pytest.raises(AttributeError):
        obj.value = 11

    # hashable
    assert {obj: 1}.get(obj) == 1

    # weakrefable
    ref = weakref.ref(obj)
    assert ref() == obj

    # serializable
    assert pickle.loads(pickle.dumps(obj)) == obj


def test_composition_of_concrete_and_singleton():
    class ConcSing(Concrete, Singleton):
        value = CoercedTo(int)

    class SingConc(Singleton, Concrete):
        value = CoercedTo(int)

    # arguments looked up after validation
    obj = ConcSing("3")
    assert ConcSing("3") is obj
    assert ConcSing(3) is obj
    assert ConcSing(3.0) is obj

    # arguments looked up before validation
    obj = SingConc("3")
    assert SingConc("3") is obj
    obj2 = SingConc(3)
    assert obj2 is not obj
    assert SingConc(3) is obj2


def test_init_subclass_keyword_arguments():
    class Test(Annotable):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()
            cls.kwargs = kwargs

    class Test2(Test, something="value", value="something"):
        pass

    assert Test2.kwargs == {"something": "value", "value": "something"}


def test_argument_order_using_optional_annotations():
    class Case1(Annotable):
        results: Optional[tuple[int]] = ()
        default: Optional[int] = None

    class SimpleCase1(Case1):
        base: int
        cases: Optional[tuple[int]] = ()

    class Case2(Annotable):
        results = optional(TupleOf(is_int), default=())
        default = optional(is_int)

    class SimpleCase2(Case1):
        base = is_int
        cases = optional(TupleOf(is_int), default=())

    assert (
        SimpleCase1.__argnames__
        == SimpleCase2.__argnames__
        == ("base", "cases", "results", "default")
    )


def test_annotable_with_optional_coercible_typehint():
    class Example(Annotable):
        value: Optional[MyInt] = None

    assert Example().value is None
    assert Example(None).value is None
    assert Example(1).value == 1
    assert isinstance(Example(1).value, MyInt)


def test_error_message(snapshot):
    class Example(Annotable):
        a: int
        b: int = 0
        c: str = "foo"
        d: Optional[float] = None
        e: tuple[int, ...] = (1, 2, 3)
        f: As[int] = 1

    with pytest.raises(ValidationError) as exc_info:
        Example("1", "2", "3", "4", "5", [])

    # assert "Failed" in str(exc_info.value)

    if sys.version_info >= (3, 11):
        target = "error_message_py311.txt"
    else:
        target = "error_message.txt"
    snapshot.assert_match(str(exc_info.value), target)
