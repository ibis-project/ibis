import pickle
import weakref

import pytest

from ibis.common.annotations import (
    Attribute,
    Parameter,
    Signature,
    immutable_property,
    initialized,
    mandatory,
    optional,
    variadic,
)
from ibis.common.caching import WeakCache
from ibis.common.graph import Traversable
from ibis.common.grounds import (
    Annotable,
    Base,
    Comparable,
    Concrete,
    Immutable,
    Singleton,
)
from ibis.common.validators import instance_of, validator
from ibis.tests.util import assert_pickle_roundtrip
from ibis.util import frozendict

is_any = instance_of(object)
is_bool = instance_of(bool)
is_float = instance_of(float)
is_int = instance_of(int)
is_str = instance_of(str)
is_list = instance_of(list)


class Op(Annotable):
    pass


class Value(Op):
    arg = instance_of(object)


class StringOp(Value):
    arg = instance_of(str)


def test_annotable():
    class Between(Annotable):
        value = is_int
        lower = optional(is_int, default=0)
        upper = optional(is_int, default=None)

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
    assert obj == obj.copy()

    obj2 = InBetween(10, lower=8)
    assert obj.copy(lower=8) == obj2


def test_variadic_annotable():
    class Test(Annotable):
        value = is_int
        rest = variadic(is_int)

    t = Test(1, 2, 3, 4)
    assert t.value == 1
    assert t.rest == (2, 3, 4)

    class Test2(Test):
        option = is_str

    with pytest.raises(TypeError):
        Test2(1, 2, 3, 4, 'foo')

    t2 = Test2(1, 2, 3, 4, option='foo')
    assert t2.value == 1
    assert t2.rest == (2, 3, 4)
    assert t2.option == 'foo'

    assert t2 == t2.copy()

    t3 = t2.copy(rest=(5, 6, 7, 8))
    assert t3 == Test2(1, 5, 6, 7, 8, option='foo')


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


def test_annotable_with_type_annotations():
    # TODO(kszucs): bar: str = None  # should raise
    class Op(Annotable):
        foo: int
        bar: str = ""

    p = Op(1)
    assert p.foo == 1
    assert p.bar == ""

    class Op(Annotable):
        bar: str = None

    with pytest.raises(TypeError):
        Op()


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
    with pytest.raises(TypeError):
        obj.value = 1

    obj = ImmAnn(3, lower=0, upper=4)
    with pytest.raises(TypeError):
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
    assert param_names == ['value', 'lower', 'upper']


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
            Parameter('left', annotation=mandatory(is_int)),
            Parameter('right', annotation=mandatory(is_int)),
        ]
    )

    assert FloatAddRhs.__signature__ == Signature(
        [
            Parameter('left', annotation=mandatory(is_int)),
            Parameter('right', annotation=mandatory(is_float)),
        ]
    )

    assert FloatAddClip.__signature__ == Signature(
        [
            Parameter('left', annotation=mandatory(is_float)),
            Parameter('right', annotation=mandatory(is_float)),
            Parameter('clip_lower', annotation=optional(is_int, default=0)),
            Parameter('clip_upper', annotation=optional(is_int, default=10)),
        ]
    )

    assert IntAddClip.__signature__ == Signature(
        [
            Parameter('left', annotation=mandatory(is_int)),
            Parameter('right', annotation=mandatory(is_int)),
            Parameter('clip_lower', annotation=optional(is_int, default=0)),
            Parameter('clip_upper', annotation=optional(is_int, default=10)),
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


def test_not_copy_default():
    default = tuple()

    class Op(Annotable):
        arg = optional(instance_of(tuple), default=default)

    op = Op()
    assert op.arg is default


def test_slots_are_inherited_and_overridable():
    class Op(Annotable):
        __slots__ = ('_cache',)  # first definition
        arg = validator(lambda x: x)

    class StringOp(Op):
        arg = validator(str)  # new overridden slot

    class StringSplit(StringOp):
        sep = validator(str)  # new slot

    class StringJoin(StringOp):
        __slots__ = ('_memoize',)  # new slot
        sep = validator(str)  # new overridden slot

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
        arg = instance_of(object)

    class Reduction(Value):
        pass

    class UDF(Value):
        func = validator(lambda fn, this: fn)

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

    assert UDAF.__slots__ == ('arity',)
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

    assert (
        str(Sum.__signature__)
        == "(arg: instance_of(<class 'object'>,), version: instance_of(<class 'int'>,), where: option(instance_of(<class 'bool'>,),default=False) = None)"  # noqa: E501
    )


def test_multiple_inheritance_optional_argument_order():
    class Value(Annotable):
        pass

    class ConditionalOp(Annotable):
        where = optional(is_bool, default=False)

    class Between(Value, ConditionalOp):
        min = is_int
        max = is_int
        how = optional(is_str, default="strict")

    assert (
        str(Between.__signature__)
        == "(min: instance_of(<class 'int'>,), max: instance_of(<class 'int'>,), how: option(instance_of(<class 'str'>,),default='strict') = None, where: option(instance_of(<class 'bool'>,),default=False) = None)"  # noqa: E501
    )


def test_immutability():
    class Value(Annotable, Immutable):
        a = is_int

    op = Value(1)
    with pytest.raises(TypeError):
        op.a = 3


class Value(Annotable):
    i = is_int
    j = Attribute(is_int)


def test_annotable_attribute():
    with pytest.raises(TypeError, match="too many positional arguments"):
        Value(1, 2)

    v = Value(1)
    assert v.__slots__ == ('i', 'j')
    assert v.i == 1
    assert not hasattr(v, 'j')
    v.j = 2
    assert v.j == 2

    with pytest.raises(TypeError):
        v.j = 'foo'


def test_annotable_mutability_and_serialization():
    v_ = Value(1)
    v = Value(1)
    assert v_ == v

    assert repr(v) == "Value(i=1)"
    assert v.__getstate__() == {'i': 1, 'j': None}
    w = pickle.loads(pickle.dumps(v))
    assert v == w

    v.j = 4
    assert v_ != v
    assert v.__getstate__() == {'i': 1, 'j': 4}
    w = pickle.loads(pickle.dumps(v))
    assert w == v
    assert repr(w) == "Value(i=1)"


def test_initialized_attribute_basics():
    class Value(Annotable):
        a = is_int

        @initialized
        def double_a(self):
            return 2 * self.a

    op = Value(1)
    assert op.a == 1
    assert op.double_a == 2
    assert len(Value.__attributes__) == 2
    assert "double_a" in Value.__slots__

    assert initialized is immutable_property


def test_initialized_attribute_mixed_with_classvar():
    class Value(Annotable):
        arg = is_int

        output_shape = "like-arg"
        output_dtype = "like-arg"

    class Reduction(Value):
        output_shape = "scalar"

    class variadic(Value):
        @initialized
        def output_shape(self):
            if self.arg > 10:
                return "columnar"
            else:
                return "scalar"

    r = Reduction(1)
    assert r.output_shape == "scalar"
    assert "output_shape" not in r.__slots__

    v = variadic(1)
    assert v.output_shape == "scalar"
    assert "output_shape" in v.__slots__

    v = variadic(100)
    assert v.output_shape == "columnar"
    assert "output_shape" in v.__slots__


class Node(Comparable):

    # override the default cache object
    __cache__ = WeakCache()
    __slots__ = ('name',)
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
    __slots__ = ('nullable',)
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
        value = validator(lambda x, this: int(x))

    class SingAnn(Singleton, Annotable):
        # this is the preferable method resolution order
        value = validator(lambda x, this: int(x))

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


class Between(Concrete):
    value = is_int
    lower = optional(is_int, default=0)
    upper = optional(is_int, default=None)

    @immutable_property
    def calculated(self):
        return self.value + self.lower


def test_concrete():
    assert Between.__mro__ == (
        Between,
        Concrete,
        Immutable,
        Comparable,
        Annotable,
        Base,
        Traversable,
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

    # serializable
    assert obj.__getstate__() == {"value": 10, "lower": 5, "upper": 15}
    assert pickle.loads(pickle.dumps(obj)) == obj


def test_concrete_with_traversable_children():
    class Bool(Concrete):
        pass

    class Value(Bool):
        value = is_bool

    class Either(Bool):
        left = instance_of(Bool)
        right = instance_of(Bool)

    class All(Bool):
        arguments = variadic(instance_of(Bool))
        strict = is_bool

    T, F = Value(True), Value(False)

    node = All(T, F, strict=True)
    assert node.__args__ == ((T, F), True)
    assert node.__children__ == (T, F)

    node = Either(T, F)
    assert node.__args__ == (T, F)
    assert node.__children__ == (T, F)

    node = All(T, Either(T, Either(T, F)), strict=False)
    assert node.__args__ == ((T, Either(T, Either(T, F))), False)
    assert node.__children__ == (T, Either(T, Either(T, F)))

    copied = node.copy(arguments=(T, F))
    assert copied == All(T, F, strict=False)


def test_composition_of_concrete_and_singleton():
    class ConcSing(Concrete, Singleton):
        value = validator(lambda x, this: int(x))

    class SingConc(Singleton, Concrete):
        value = validator(lambda x, this: int(x))

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


# TODO(kszucs): test that annotable subclasses can use __init_subclass__ kwargs
