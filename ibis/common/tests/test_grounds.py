import weakref
from inspect import Signature

import pytest

from ibis.common.caching import WeakCache
from ibis.common.grounds import (
    Annotable,
    Base,
    Comparable,
    Concrete,
    Immutable,
    Parameter,
    Singleton,
)
from ibis.common.validators import Optional, Validator, immutable_property
from ibis.tests.util import assert_pickle_roundtrip
from ibis.util import frozendict


class ValidatorFunction(Validator):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class InstanceOf(Validator):
    def __init__(self, typ):
        self.typ = typ

    def __call__(self, arg, **kwargs):
        if not isinstance(arg, self.typ):
            raise TypeError(self.typ)
        return arg


IsAny = InstanceOf(object)
IsBool = InstanceOf(bool)
IsFloat = InstanceOf(float)
IsInt = InstanceOf(int)
IsStr = InstanceOf(str)


class Op(Annotable):
    pass


class Value(Op):
    arg = InstanceOf(object)


class StringOp(Value):
    arg = InstanceOf(str)


def test_annotable():
    class Between(Annotable):
        value = IsInt
        lower = Optional(IsInt, default=0)
        upper = Optional(IsInt, default=None)

    class InBetween(Between):
        pass

    argnames = ('value', 'lower', 'upper')
    signature = Between.__signature__
    assert isinstance(signature, Signature)
    assert tuple(signature.parameters.keys()) == argnames
    assert Between.__slots__ == argnames

    obj = Between(10, lower=2)
    assert obj.value == 10
    assert obj.lower == 2
    assert obj.upper is None
    assert obj.__argnames__ == argnames
    assert obj.__slots__ == ("value", "lower", "upper")
    assert not hasattr(obj, "__dict__")

    # test that a child without additional arguments doesn't have __dict__
    obj = InBetween(10, lower=2)
    assert obj.__slots__ == tuple()
    assert not hasattr(obj, "__dict__")


def test_annotable_is_mutable_by_default():
    # TODO(kszucs): more exhaustive testing of mutability, e.g. setting
    # optional value to None doesn't set to the default value
    class Op(Annotable):
        a = IsInt
        b = IsInt

    p = Op(1, 2)
    assert p.a == 1
    p.a = 3
    assert p.a == 3
    assert p == Op(a=3, b=2)


def test_composition_of_annotable_and_immutable():
    class AnnImm(Annotable, Immutable):
        value = IsInt
        lower = Optional(IsInt, default=0)
        upper = Optional(IsInt, default=None)

    class ImmAnn(Immutable, Annotable):
        # this is the preferable method resolution order
        value = IsInt
        lower = Optional(IsInt, default=0)
        upper = Optional(IsInt, default=None)

    obj = AnnImm(3, lower=0, upper=4)
    with pytest.raises(TypeError):
        obj.value = 1

    obj = ImmAnn(3, lower=0, upper=4)
    with pytest.raises(TypeError):
        obj.value = 1


def test_composition_of_annotable_and_comparable():
    class Between(Comparable, Annotable):
        value = IsInt
        lower = Optional(IsInt, default=0)
        upper = Optional(IsInt, default=None)

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
        value = IsInt
        lower = Optional(IsInt, default=0)
        upper = Optional(IsInt, default=None)

    param_names = list(Between.__signature__.parameters.keys())
    assert param_names == ['value', 'lower', 'upper']


def test_signature_inheritance():
    class IntBinop(Annotable):
        left = IsInt
        right = IsInt

    class FloatAddRhs(IntBinop):
        right = IsFloat

    class FloatAddClip(FloatAddRhs):
        left = IsFloat
        clip_lower = Optional(IsInt, default=0)
        clip_upper = Optional(IsInt, default=10)

    class IntAddClip(FloatAddClip, IntBinop):
        pass

    assert IntBinop.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt),
            Parameter('right', validator=IsInt),
        ]
    )

    assert FloatAddRhs.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt),
            Parameter('right', validator=IsFloat),
        ]
    )

    assert FloatAddClip.__signature__ == Signature(
        [
            Parameter('left', validator=IsFloat),
            Parameter('right', validator=IsFloat),
            Parameter('clip_lower', validator=Optional(IsInt, default=0)),
            Parameter('clip_upper', validator=Optional(IsInt, default=10)),
        ]
    )

    assert IntAddClip.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt),
            Parameter('right', validator=IsInt),
            Parameter('clip_lower', validator=Optional(IsInt, default=0)),
            Parameter('clip_upper', validator=Optional(IsInt, default=10)),
        ]
    )


def test_positional_argument_reordering():
    class Farm(Annotable):
        ducks = IsInt
        donkeys = IsInt
        horses = IsInt
        goats = IsInt
        chickens = IsInt

    class NoHooves(Farm):
        horses = Optional(IsInt, default=0)
        goats = Optional(IsInt, default=0)
        donkeys = Optional(IsInt, default=0)

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
        a = IsInt
        b = IsInt

    class Beta(Alpha):
        c = IsInt
        d = Optional(IsInt, default=0)
        e = IsInt

    obj = Beta(1, 2, 3, 4)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.e == 4
    assert obj.d == 0

    obj = Beta(1, 2, 3, 4, 5)
    assert obj.d == 5
    assert obj.e == 4


def test_not_copy_default():
    default = tuple()

    class Op(Annotable):
        arg = Optional(InstanceOf(tuple), default=default)

    op = Op()
    assert op.arg is default


def test_slots_are_inherited_and_overridable():
    class Op(Annotable):
        __slots__ = ('_cache',)  # first definition
        arg = ValidatorFunction(lambda x: x)

    class StringOp(Op):
        arg = ValidatorFunction(str)  # new overridden slot

    class StringSplit(StringOp):
        sep = ValidatorFunction(str)  # new slot

    class StringJoin(StringOp):
        __slots__ = ('_memoize',)  # new slot
        sep = ValidatorFunction(str)  # new overridden slot

    assert Op.__slots__ == ('_cache', 'arg')
    assert StringOp.__slots__ == ('arg',)
    assert StringSplit.__slots__ == ('sep',)
    assert StringJoin.__slots__ == ('_memoize', 'sep')


def test_multiple_inheritance():
    # multiple inheritance is allowed only if one of the parents has non-empty
    # __slots__ definition, otherwise python will raise lay-out conflict

    class Op(Annotable):
        __slots__ = ('_hash',)

    class Value(Annotable):
        arg = InstanceOf(object)

    class Reduction(Value):
        pass

    class UDF(Value):
        func = ValidatorFunction(lambda fn, this: fn)

    class UDAF(UDF, Reduction):
        arity = IsInt

    class A(Annotable):
        a = IsInt

    class B(Annotable):
        b = IsInt

    msg = "multiple bases have instance lay-out conflict"
    with pytest.raises(TypeError, match=msg):

        class AB(A, B):
            ab = IsInt

    assert UDAF.__slots__ == ('arity',)
    strlen = UDAF(arg=2, func=lambda value: len(str(value)), arity=1)
    assert strlen.arg == 2
    assert strlen.arity == 1


@pytest.mark.parametrize(
    "obj",
    [
        StringOp(arg="something"),
        Parameter("test"),
    ],
)
def test_pickling_support(obj):
    assert_pickle_roundtrip(obj)


def test_multiple_inheritance_argument_order():
    class Value(Annotable):
        arg = IsAny

    class VersionedOp(Value):
        version = IsInt

    class Reduction(Annotable):
        pass

    class Sum(VersionedOp, Reduction):
        where = Optional(IsBool, default=False)

    assert str(Sum.__signature__) == "(arg, version, where=None)"


def test_multiple_inheritance_optional_argument_order():
    class Value(Annotable):
        pass

    class ConditionalOp(Annotable):
        where = Optional(IsBool, default=False)

    class Between(Value, ConditionalOp):
        min = IsInt
        max = IsInt
        how = Optional(IsStr, default="strict")

    assert str(Between.__signature__) == "(min, max, how=None, where=None)"


def test_immutability():
    class Value(Annotable, Immutable):
        a = IsInt

    op = Value(1)
    with pytest.raises(TypeError):
        op.a = 3


def test_immutable_property_basics():
    class Value(Annotable):
        a = IsInt

        @immutable_property
        def double_a(self):
            return 2 * self.a

    op = Value(1)
    assert op.a == 1
    assert op.double_a == 2
    assert len(Value.__properties__) == 1
    assert "double_a" in Value.__slots__


def test_immutable_property_mixed_with_classvar():
    class Value(Annotable):
        arg = IsInt

        output_shape = "like-arg"
        output_dtype = "like-arg"

    class Reduction(Value):
        output_shape = "scalar"

    class Variadic(Value):
        @immutable_property
        def output_shape(self):
            if self.arg > 10:
                return "columnar"
            else:
                return "scalar"

    r = Reduction(1)
    assert r.output_shape == "scalar"
    assert "output_shape" not in r.__slots__

    v = Variadic(1)
    assert v.output_shape == "scalar"
    assert "output_shape" in v.__slots__

    v = Variadic(100)
    assert v.output_shape == "columnar"
    assert "output_shape" in v.__slots__


class Node(Comparable):

    # override the default cache object
    __cache__ = WeakCache()
    num_equal_calls = 0

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Node(name={self.name})"

    def __equals__(self, other):
        Node.num_equal_calls += 1
        return self.name == other.name


@pytest.fixture
def cache():
    Node.num_equal_calls = 0
    cache = Node.__cache__
    try:
        yield cache
    finally:
        assert len(cache) == 0


def pair(a, b):
    # for same ordering with comparable
    if id(a) < id(b):
        return (a, b)
    else:
        return (b, a)


def test_comparable_basic(cache):
    a = Node(name="a")
    b = Node(name="a")
    c = Node(name="a")
    assert a == b
    assert a == c
    del a
    del b
    del c


def test_comparable_caching(cache):
    a = Node(name="a")
    b = Node(name="b")
    c = Node(name="c")
    d = Node(name="d")
    e = Node(name="e")

    cache[pair(a, b)] = True
    cache[pair(a, c)] = False
    cache[pair(c, d)] = True
    cache[pair(b, d)] = False
    assert len(cache) == 4

    assert a == b
    assert a != c
    assert c == d
    assert b != d
    assert Node.num_equal_calls == 0

    # no cache hit
    assert pair(a, e) not in cache
    assert a != e
    assert Node.num_equal_calls == 1
    assert len(cache) == 5

    # run only once
    assert e != a
    assert Node.num_equal_calls == 1
    assert pair(a, e) in cache


def test_comparable_garbage_collection(cache):
    a = Node(name="a")
    b = Node(name="b")
    c = Node(name="c")
    d = Node(name="d")

    cache[pair(a, b)] = True
    cache[pair(a, c)] = False
    cache[pair(c, d)] = True
    cache[pair(b, d)] = False

    assert weakref.getweakrefcount(a) == 2
    del c
    assert weakref.getweakrefcount(a) == 1
    del b
    assert weakref.getweakrefcount(a) == 0


def test_comparable_cache_reuse(cache):
    nodes = [
        Node(name="a"),
        Node(name="b"),
        Node(name="c"),
        Node(name="d"),
        Node(name="e"),
    ]

    expected = 0
    for a, b in zip(nodes, nodes):
        a == a
        a == b
        b == a
        if a != b:
            expected += 1
        assert Node.num_equal_calls == expected

    assert len(cache) == expected

    # check that cache is evicted once nodes get collected
    del nodes
    assert len(cache) == 0

    a = Node(name="a")
    b = Node(name="a")
    assert a == b


class OneAndOnly(Singleton):
    __instances__ = weakref.WeakValueDictionary()


class DataType(Singleton):
    __instances__ = weakref.WeakValueDictionary()

    def __init__(self, nullable=True):
        self.nullable = nullable


def test_singleton_basics():
    one = OneAndOnly()
    only = OneAndOnly()
    assert one is only

    assert len(OneAndOnly.__instances__) == 1
    key = (OneAndOnly, (), frozendict())
    assert OneAndOnly.__instances__[key] is one


def test_singleton_lifetime():
    one = OneAndOnly()
    assert len(OneAndOnly.__instances__) == 1

    del one
    assert len(OneAndOnly.__instances__) == 0


def test_singleton_with_argument():
    dt1 = DataType(nullable=True)
    dt2 = DataType(nullable=False)
    dt3 = DataType(nullable=True)

    assert dt1 is dt3
    assert dt1 is not dt2
    assert len(DataType.__instances__) == 2

    del dt3
    assert len(DataType.__instances__) == 2
    del dt1
    assert len(DataType.__instances__) == 1
    del dt2
    assert len(DataType.__instances__) == 0


def test_composition_of_annotable_and_singleton():
    class AnnSing(Annotable, Singleton):
        value = ValidatorFunction(lambda x, this: int(x))

    class SingAnn(Singleton, Annotable):
        # this is the preferable method resolution order
        value = ValidatorFunction(lambda x, this: int(x))

    # arguments looked up after validation
    obj = AnnSing("3")
    assert AnnSing("3") is obj
    assert AnnSing(3) is obj
    assert AnnSing(3.0) is obj

    # arguments looked up before validation
    obj = SingAnn("3")
    assert SingAnn("3") is obj
    obj2 = SingAnn(3)
    assert obj2 is not obj
    assert SingAnn(3) is obj2


def test_concrete():
    class Between(Concrete):
        value = IsInt
        lower = Optional(IsInt, default=0)
        upper = Optional(IsInt, default=None)

        @immutable_property
        def calculated(self):
            return self.value + self.lower

    assert Between.__mro__ == (
        Between,
        Concrete,
        Immutable,
        Comparable,
        Annotable,
        Base,
        object,
    )

    assert Between.__create__.__func__ is Annotable.__create__.__func__
    assert Between.__eq__ is Comparable.__eq__
    assert Between.__argnames__ == ("value", "lower", "upper")

    # annotable
    obj = Between(10, lower=5, upper=15)
    obj2 = Between(10, lower=5, upper=15)
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
    with pytest.raises(TypeError):
        obj.value = 11

    # hashable
    assert {obj: 1}.get(obj) == 1

    # weakrefable
    ref = weakref.ref(obj)
    assert ref() == obj


def test_composition_of_concrete_and_singleton():
    class ConcSing(Concrete, Singleton):
        value = ValidatorFunction(lambda x, this: int(x))

    class SingConc(Singleton, Concrete):
        value = ValidatorFunction(lambda x, this: int(x))

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
