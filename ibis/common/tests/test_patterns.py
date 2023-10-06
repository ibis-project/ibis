from __future__ import annotations

import functools
import re
import sys
from collections.abc import Callable as CallableABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (  # noqa: UP035
    Annotated,
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)
from typing import (
    Any as AnyType,
)

import pytest

from ibis.common.annotations import ValidationError
from ibis.common.collections import FrozenDict
from ibis.common.deferred import Call, deferred, var
from ibis.common.graph import Node as GraphNode
from ibis.common.patterns import (
    AllOf,
    Any,
    AnyOf,
    Between,
    CallableWith,
    Capture,
    Check,
    CoercedTo,
    Coercible,
    Contains,
    Custom,
    DictOf,
    EqualTo,
    FrozenDictOf,
    GenericInstanceOf,
    GenericSequenceOf,
    InstanceOf,
    IsIn,
    LazyInstanceOf,
    Length,
    ListOf,
    MappingOf,
    Node,
    NoMatch,
    NoneOf,
    Not,
    Nothing,
    Object,
    Option,
    Pattern,
    PatternMapping,
    PatternSequence,
    Replace,
    SequenceOf,
    SubclassOf,
    TupleOf,
    TypeOf,
    Variable,
    _,
    match,
    pattern,
    replace,
)
from ibis.util import Namespace


class Double(Pattern):
    def match(self, value, *, context):
        return value * 2

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


class Min(Pattern):
    __slots__ = ("min",)

    def __init__(self, min):
        self.min = min

    def match(self, value, context):
        if value >= self.min:
            return value
        else:
            return NoMatch

    def __hash__(self):
        return hash((self.__class__, self.min))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.min == other.min


x = Variable("x")
y = Variable("y")
z = Variable("z")


def test_immutability_of_patterns():
    p = InstanceOf(int)
    with pytest.raises(AttributeError):
        p.types = [str]


def test_nothing():
    p = Nothing()
    assert p.match(1, context={}) is NoMatch
    assert p.match(2, context={}) is NoMatch


def test_min():
    p = Min(10)
    assert p.match(10, context={}) == 10
    assert p.match(9, context={}) is NoMatch


def test_double():
    p = Double()
    assert p.match(10, context={}) == 20


def test_any():
    p = Any()
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) == "foo"
    assert p.describe() == "matching Any()"


def test_pattern_factory_wraps_variable_with_capture():
    v = var("other")
    p = pattern(v)
    assert p == Capture("other", Any())

    ctx = {}
    assert p.match(10, ctx) == 10
    assert ctx == {"other": 10}


def test_capture():
    ctx = {}

    p = Capture("result", Min(11))
    assert p.match(10, context=ctx) is NoMatch
    assert ctx == {}

    assert p.match(12, context=ctx) == 12
    assert ctx == {"result": 12}


def test_option():
    p = Option(InstanceOf(str))
    assert Option(str) == p
    assert p.match(None, context={}) is None
    assert p.match("foo", context={}) == "foo"
    assert p.match(1, context={}) is NoMatch
    assert p.describe() == "either None or a str"
    assert p.describe(plural=True) == "optional strs"

    p = Option(int, default=-1)
    assert p.match(None, context={}) == -1
    assert p.match(1, context={}) == 1
    assert p.match(1.0, context={}) is NoMatch
    assert p.describe() == "either None or an int"
    assert p.describe(plural=True) == "optional ints"


def test_check():
    def checker(x):
        return x == 10

    p = Check(checker)
    assert p.match(10, context={}) == 10
    assert p.match(11, context={}) is NoMatch
    assert p.describe() == "a value that satisfies checker()"
    assert p.describe(plural=True) == "values that satisfy checker()"


def test_equal_to():
    p = EqualTo(10)
    assert p.match(10, context={}) == 10
    assert p.match(11, context={}) is NoMatch
    assert p.describe() == "10"
    assert p.describe(plural=True) == "10"

    p = EqualTo("10")
    assert p.match(10, context={}) is NoMatch
    assert p.match("10", context={}) == "10"
    assert p.describe() == "'10'"
    assert p.describe(plural=True) == "'10'"


def test_type_of():
    p = TypeOf(int)
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) is NoMatch
    assert p.describe() == "exactly an int"
    assert p.describe(plural=True) == "exactly ints"


def test_subclass_of():
    p = SubclassOf(Pattern)
    assert p.match(Double, context={}) == Double
    assert p.match(int, context={}) is NoMatch
    assert p.describe() == "a subclass of Pattern"
    assert p.describe(plural=True) == "subclasses of Pattern"


def test_instance_of():
    p = InstanceOf(int)
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) is NoMatch
    assert p.describe() == "an int"
    assert p.describe(plural=True) == "ints"

    p = InstanceOf((int, str))
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) == "foo"
    assert p.match(1.0, context={}) is NoMatch
    assert p.describe() == "an int or a str"
    assert p.describe(plural=True) == "ints or strs"

    p = InstanceOf((int, str, float))
    assert p.describe() == "an int, a str or a float"


def test_lazy_instance_of():
    p = LazyInstanceOf("re.Pattern")
    assert p.match(re.compile("foo"), context={}) == re.compile("foo")
    assert p.match("foo", context={}) is NoMatch


T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)


@dataclass
class My(Generic[T, S]):
    a: T
    b: S
    c: str


def test_generic_instance_of_with_covariant_typevar():
    p = Pattern.from_typehint(My[int, AnyType])
    assert p.match(My(1, 2, "3"), context={}) == My(1, 2, "3")
    assert p.describe() == "a My[int, Any]"

    assert match(My[int, AnyType], v := My(1, 2, "3")) == v
    assert match(My[int, int], v := My(1, 2, "3")) == v
    assert match(My[int, float], My(1, 2, "3")) is NoMatch
    assert match(My[int, float], v := My(1, 2.0, "3")) == v


def test_generic_instance_of_disallow_nested_coercion():
    class MyString(str, Coercible):
        @classmethod
        def __coerce__(cls, other):
            return cls(str(other))

    class Box(Generic[T]):
        value: T

    p = Pattern.from_typehint(Box[MyString])
    assert isinstance(p, GenericInstanceOf)
    assert p.origin == Box
    assert p.fields == {"value": InstanceOf(MyString)}


def test_coerced_to():
    class MyInt(int, Coercible):
        @classmethod
        def __coerce__(cls, other):
            return MyInt(MyInt(other) + 1)

    p = CoercedTo(int)
    assert p.match(1, context={}) == 1
    assert p.match("1", context={}) == 1
    with pytest.raises(ValueError):
        p.match("foo", context={})

    p = CoercedTo(MyInt)
    assert p.match(1, context={}) == 2
    assert p.match("1", context={}) == 2
    with pytest.raises(ValueError):
        p.match("foo", context={})


def test_generic_coerced_to():
    class DataType:
        def __eq__(self, other):
            return type(self) == type(other)

    class Integer(DataType):
        pass

    class String(DataType):
        pass

    class DataShape:
        def __eq__(self, other):
            return type(self) == type(other)

    class Scalar(DataShape):
        pass

    class Array(DataShape):
        pass

    class Value(Generic[T, S], Coercible):
        @classmethod
        def __coerce__(cls, value, T=..., S=...):
            return cls(value, Scalar())

        def dtype(self) -> T:
            ...

        def shape(self) -> S:
            ...

    class Literal(Value[T, Scalar]):
        __slots__ = ("_value", "_dtype")

        def __init__(self, value, dtype):
            self._value = value
            self._dtype = dtype

        def dtype(self) -> T:
            return self.dtype

        def shape(self) -> DataShape:
            return Scalar()

        def __eq__(self, other):
            return (
                type(self) == type(other)
                and self._value == other._value
                and self._dtype == other._dtype
            )

    p = Pattern.from_typehint(Literal[String])
    r = p.match("foo", context={})
    assert r == Literal("foo", Scalar())
    expected = "coercible to a <locals>.Literal[<locals>.String]"
    assert p.describe() == expected


def test_not():
    p = Not(InstanceOf(int))
    p1 = ~InstanceOf(int)

    assert p == p1
    assert p.match(1, context={}) is NoMatch
    assert p.match("foo", context={}) == "foo"
    assert p.describe() == "anything except an int"
    assert p.describe(plural=True) == "anything except ints"


def test_any_of():
    p = AnyOf(InstanceOf(int), InstanceOf(str))
    p1 = InstanceOf(int) | InstanceOf(str)

    assert p == p1
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) == "foo"
    assert p.match(1.0, context={}) is NoMatch
    assert p.describe() == "an int or a str"
    assert p.describe(plural=True) == "ints or strs"

    p = AnyOf(InstanceOf(int), InstanceOf(str), InstanceOf(float))
    assert p.describe() == "an int, a str or a float"


def test_all_of():
    def negative(x):
        return x < 0

    p = AllOf(InstanceOf(int), Check(negative))
    p1 = InstanceOf(int) & Check(negative)

    assert p == p1
    assert p.match(1, context={}) is NoMatch
    assert p.match(-1, context={}) == -1
    assert p.match(1.0, context={}) is NoMatch
    assert p.describe() == "an int then a value that satisfies negative()"

    p = AllOf(InstanceOf(int), CoercedTo(float), CoercedTo(str))
    assert p.match(1, context={}) == "1.0"
    assert p.match(1.0, context={}) is NoMatch
    assert p.match("1", context={}) is NoMatch
    assert p.describe() == "an int, coercible to a float then coercible to a str"


def test_none_of():
    def negative(x):
        return x < 0

    p = NoneOf(InstanceOf(int), Check(negative))
    assert p.match(1.0, context={}) == 1.0
    assert p.match(-1.0, context={}) is NoMatch
    assert p.match(1, context={}) is NoMatch
    assert p.describe() == "anything except an int or a value that satisfies negative()"


def test_length():
    with pytest.raises(ValueError):
        Length(exactly=3, at_least=3)
    with pytest.raises(ValueError):
        Length(exactly=3, at_most=3)

    p = Length(exactly=3)
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([1, 2], context={}) is NoMatch
    assert p.describe() == "with length exactly 3"

    p = Length(at_least=3)
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([1, 2], context={}) is NoMatch
    assert p.describe() == "with length at least 3"

    p = Length(at_most=3)
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([1, 2, 3, 4], context={}) is NoMatch
    assert p.describe() == "with length at most 3"

    p = Length(at_least=3, at_most=5)
    assert p.match([1, 2], context={}) is NoMatch
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([1, 2, 3, 4], context={}) == [1, 2, 3, 4]
    assert p.match([1, 2, 3, 4, 5], context={}) == [1, 2, 3, 4, 5]
    assert p.match([1, 2, 3, 4, 5, 6], context={}) is NoMatch
    assert p.describe() == "with length between 3 and 5"


def test_contains():
    p = Contains(1)
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([2, 3], context={}) is NoMatch
    assert p.match({1, 2, 3}, context={}) == {1, 2, 3}
    assert p.match({2, 3}, context={}) is NoMatch
    assert p.describe() == "containing 1"
    assert p.describe(plural=True) == "containing 1"

    p = Contains("1")
    assert p.match([1, 2, 3], context={}) is NoMatch
    assert p.match(["1", 2, 3], context={}) == ["1", 2, 3]
    assert p.match("123", context={}) == "123"
    assert p.describe() == "containing '1'"


def test_isin():
    p = IsIn([1, 2, 3])
    assert p.match(1, context={}) == 1
    assert p.match(4, context={}) is NoMatch
    assert p.describe() == "in {1, 2, 3}"
    assert p.describe(plural=True) == "in {1, 2, 3}"


def test_sequence_of():
    p = SequenceOf(InstanceOf(str), list)
    assert isinstance(p, SequenceOf)
    assert p.match(["foo", "bar"], context={}) == ["foo", "bar"]
    assert p.match([1, 2], context={}) is NoMatch
    assert p.match(1, context={}) is NoMatch
    assert p.match("string", context={}) is NoMatch
    assert p.describe() == "a list of strs"
    assert p.describe(plural=True) == "lists of strs"


def test_generic_sequence_of():
    class MyList(list, Coercible):
        @classmethod
        def __coerce__(cls, value, T=...):
            return cls(value)

    p = SequenceOf(InstanceOf(str), MyList)
    assert isinstance(p, GenericSequenceOf)
    assert p == GenericSequenceOf(InstanceOf(str), MyList)
    assert p.match(["foo", "bar"], context={}) == MyList(["foo", "bar"])
    assert p.match("string", context={}) is NoMatch

    p = SequenceOf(InstanceOf(str), tuple, at_least=1)
    assert isinstance(p, GenericSequenceOf)
    assert p == GenericSequenceOf(InstanceOf(str), tuple, at_least=1)
    assert p.match(("foo", "bar"), context={}) == ("foo", "bar")
    assert p.match([], context={}) is NoMatch

    p = GenericSequenceOf(InstanceOf(str), list)
    assert isinstance(p, SequenceOf)
    assert p == SequenceOf(InstanceOf(str), list)
    assert p.match(("foo", "bar"), context={}) == ["foo", "bar"]


def test_list_of():
    p = ListOf(InstanceOf(str))
    assert isinstance(p, SequenceOf)
    assert p.match(["foo", "bar"], context={}) == ["foo", "bar"]
    assert p.match([1, 2], context={}) is NoMatch
    assert p.match(1, context={}) is NoMatch
    assert p.describe() == "a list of strs"
    assert p.describe(plural=True) == "lists of strs"


def test_tuple_of():
    p = TupleOf((InstanceOf(str), InstanceOf(int), InstanceOf(float)))
    assert p.match(("foo", 1, 1.0), context={}) == ("foo", 1, 1.0)
    assert p.match(["foo", 1, 1.0], context={}) == ("foo", 1, 1.0)
    assert p.match(1, context={}) is NoMatch
    assert p.describe() == "a tuple of (a str, an int, a float)"
    assert p.describe(plural=True) == "tuples of (a str, an int, a float)"

    p = TupleOf(InstanceOf(str))
    assert p == SequenceOf(InstanceOf(str), tuple)
    assert p.match(("foo", "bar"), context={}) == ("foo", "bar")
    assert p.match(["foo"], context={}) == ("foo",)
    assert p.match(1, context={}) is NoMatch
    assert p.describe() == "a tuple of strs"
    assert p.describe(plural=True) == "tuples of strs"


def test_mapping_of():
    p = MappingOf(InstanceOf(str), InstanceOf(int))
    assert p.match({"foo": 1, "bar": 2}, context={}) == {"foo": 1, "bar": 2}
    assert p.match({"foo": 1, "bar": "baz"}, context={}) is NoMatch
    assert p.match(1, context={}) is NoMatch

    p = MappingOf(InstanceOf(str), InstanceOf(str), FrozenDict)
    assert p.match({"foo": "bar"}, context={}) == FrozenDict({"foo": "bar"})
    assert p.match({"foo": 1}, context={}) is NoMatch


class Foo:
    __match_args__ = ("a", "b")

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return type(self) == type(other) and self.a == other.a and self.b == other.b


class Bar:
    __match_args__ = ("c", "d")

    def __init__(self, c, d):
        self.c = c
        self.d = d

    def __eq__(self, other):
        return type(self) == type(other) and self.c == other.c and self.d == other.d


def test_object_pattern():
    p = Object(Foo, 1, b=2)
    o = Foo(1, 2)
    r = match(p, o)
    assert r is o
    assert r == Foo(1, 2)


def test_object_pattern_complex_type():
    p = Object(Not(Foo), 1, 2)
    o = Bar(1, 2)

    # test that the pattern isn't changing the input object if none of
    # its arguments are changed by subpatterns
    assert match(p, o) is o
    assert match(p, Foo(1, 2)) is NoMatch
    assert match(p, Bar(1, 3)) is NoMatch

    p = Object(Not(Foo), 1, b=2)
    assert match(p, Bar(1, 2)) is NoMatch


def test_object_pattern_from_instance_of():
    class MyType:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    p = pattern(MyType)
    assert p == InstanceOf(MyType)

    p_call = p(1, 2)
    assert p_call == Object(MyType, 1, 2)


def test_object_pattern_from_coerced_to():
    class MyCoercibleType(Coercible):
        def __init__(self, a, b):
            self.a = a
            self.b = b

        @classmethod
        def __coerce__(cls, other):
            a, b = other
            return cls(a, b)

    p = CoercedTo(MyCoercibleType)
    p_call = p(1, 2)
    assert p_call == Object(MyCoercibleType, 1, 2)


def test_callable_with():
    def func(a, b):
        return str(a) + b

    def func_with_args(a, b, *args):
        return sum((a, b) + args)

    def func_with_kwargs(a, b, c=1, **kwargs):
        return str(a) + b + str(c)

    def func_with_optional_keyword_only_kwargs(a, *, c=1):
        return a + c

    def func_with_required_keyword_only_kwargs(*, c):
        return c

    p = CallableWith([InstanceOf(int), InstanceOf(str)])
    assert p.match(10, context={}) is NoMatch

    msg = "Callable has mandatory keyword-only arguments which cannot be specified"
    with pytest.raises(TypeError, match=msg):
        p.match(func_with_required_keyword_only_kwargs, context={})

    # Callable has more positional arguments than expected
    p = CallableWith([InstanceOf(int)] * 2)
    assert p.match(func_with_kwargs, context={}).__wrapped__ is func_with_kwargs

    # Callable has less positional arguments than expected
    p = CallableWith([InstanceOf(int)] * 4)
    assert p.match(func_with_kwargs, context={}) is NoMatch

    p = CallableWith([InstanceOf(int)] * 4, InstanceOf(int))
    wrapped = p.match(func_with_args, context={})
    assert wrapped(1, 2, 3, 4) == 10

    p = CallableWith([InstanceOf(int), InstanceOf(str)], InstanceOf(str))
    wrapped = p.match(func, context={})
    assert wrapped(1, "st") == "1st"

    with pytest.raises(ValidationError):
        wrapped(1, 2)

    p = CallableWith([InstanceOf(int)])
    wrapped = p.match(func_with_optional_keyword_only_kwargs, context={})
    assert wrapped(1) == 2


def test_callable_with_default_arguments():
    def f(a: int, b: str, c: str):
        return a + int(b) + int(c)

    def g(a: int, b: str, c: str = "0"):
        return a + int(b) + int(c)

    h = functools.partial(f, c="0")

    p = Pattern.from_typehint(Callable[[int, str], int])
    assert p.match(f, {}) is NoMatch
    assert p.match(g, {}).__wrapped__ == g
    assert p.match(h, {}).__wrapped__ == h


def test_pattern_list():
    p = PatternSequence([1, 2, InstanceOf(int), ...])
    assert p.match([1, 2, 3, 4, 5], context={}) == [1, 2, 3, 4, 5]
    assert p.match([1, 2, 3, 4, 5, 6], context={}) == [1, 2, 3, 4, 5, 6]
    assert p.match([1, 2, 3, 4], context={}) == [1, 2, 3, 4]
    assert p.match([1, 2, "3", 4], context={}) is NoMatch

    # subpattern is a simple pattern
    p = PatternSequence([1, 2, CoercedTo(int), ...])
    assert p.match([1, 2, 3.0, 4.0, 5.0], context={}) == [1, 2, 3, 4.0, 5.0]

    # subpattern is a sequence
    p = PatternSequence([1, 2, 3, SequenceOf(CoercedTo(int), at_least=1)])
    assert p.match([1, 2, 3, 4.0, 5.0], context={}) == [1, 2, 3, 4, 5]


def test_matching():
    assert match("foo", "foo") == "foo"
    assert match("foo", "bar") is NoMatch

    assert match(InstanceOf(int), 1) == 1
    assert match(InstanceOf(int), "foo") is NoMatch

    assert Capture("pi", InstanceOf(float)) == "pi" @ InstanceOf(float)
    assert Capture("pi", InstanceOf(float)) == "pi" @ InstanceOf(float)

    assert match(Capture("pi", InstanceOf(float)), 3.14, ctx := {}) == 3.14
    assert ctx == {"pi": 3.14}
    assert match("pi" @ InstanceOf(float), 3.14, ctx := {}) == 3.14
    assert ctx == {"pi": 3.14}

    assert match("pi" @ InstanceOf(float), 3.14, ctx := {}) == 3.14
    assert ctx == {"pi": 3.14}

    assert match(InstanceOf(int) | InstanceOf(float), 3) == 3
    assert match(InstanceOf(object) & InstanceOf(float), 3.14) == 3.14


def test_replace_passes_matched_value_as_underscore():
    class MyInt:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return self.value == other.value

    p = InstanceOf(int) >> Call(MyInt, value=_)
    assert p.match(1, context={}) == MyInt(1)


def test_replace_in_nested_object_pattern():
    # simple example using reference to replace a value
    b = Variable("b")
    p = Object(Foo, 1, b=Replace(..., b))
    f = p.match(Foo(1, 2), {"b": 3})
    assert f.a == 1
    assert f.b == 3

    # nested example using reference to replace a value
    d = Variable("d")
    p = Object(Foo, 1, b=Object(Bar, 2, d=Replace(..., d)))
    g = p.match(Foo(1, Bar(2, 3)), {"d": 4})
    assert g.b.c == 2
    assert g.b.d == 4

    # nested example using reference to replace a value with a captured value
    p = Object(
        Foo,
        1,
        b=Replace(Object(Bar, 2, d="d" @ Any()), lambda _, d: Foo(-1, b=d)),
    )
    h = p.match(Foo(1, Bar(2, 3)), {})
    assert isinstance(h, Foo)
    assert h.a == 1
    assert isinstance(h.b, Foo)
    assert h.b.b == 3

    # same example with more syntactic sugar
    o = Namespace(pattern, module=__name__)
    c = Namespace(deferred, module=__name__)

    d = Variable("d")
    p = o.Foo(1, b=o.Bar(2, d=d @ Any()) >> c.Foo(-1, b=d))
    h1 = p.match(Foo(1, Bar(2, 3)), {})
    assert isinstance(h1, Foo)
    assert h1.a == 1
    assert isinstance(h1.b, Foo)
    assert h1.b.b == 3


def test_replace_decorator():
    @replace(int)
    def sub(_):
        return _ - 1

    assert match(sub, 1) == 0
    assert match(sub, 2) == 1


def test_replace_using_deferred():
    p = Namespace(pattern, module=__name__)
    d = Namespace(deferred, module=__name__)

    x = var("x")
    y = var("y")

    pat = p.Foo(x, b=y) >> d.Foo(x, b=y)
    assert match(pat, Foo(1, 2)) == Foo(1, 2)

    pat = p.Foo(x, b=y) >> d.Foo(x, b=(y + 1) * x)
    assert match(pat, Foo(2, 3)) == Foo(2, 8)

    pat = p.Foo(x, y @ p.Bar) >> d.Foo(x, b=y.c + y.d)
    assert match(pat, Foo(1, Bar(2, 3))) == Foo(1, 5)


def test_matching_sequence_pattern():
    assert match([], []) == []
    assert match([], [1]) is NoMatch

    assert match([1, 2, 3, 4, ...], list(range(1, 9))) == list(range(1, 9))
    assert match([1, 2, 3, 4, ...], list(range(1, 3))) is NoMatch
    assert match([1, 2, 3, 4, ...], list(range(1, 5))) == list(range(1, 5))
    assert match([1, 2, 3, 4, ...], list(range(1, 6))) == list(range(1, 6))

    assert match([..., 3, 4], list(range(5))) == list(range(5))
    assert match([..., 3, 4], list(range(3))) is NoMatch

    assert match([0, 1, ..., 4], list(range(5))) == list(range(5))
    assert match([0, 1, ..., 4], list(range(4))) is NoMatch

    assert match([...], list(range(5))) == list(range(5))
    assert match([..., 2, 3, 4, ...], list(range(8))) == list(range(8))


def test_matching_sequence_with_captures():
    assert match([1, 2, 3, 4, SequenceOf(...)], v := list(range(1, 9))) == v
    assert (
        match([1, 2, 3, 4, "rest" @ SequenceOf(...)], v := list(range(1, 9)), ctx := {})
        == v
    )
    assert ctx == {"rest": (5, 6, 7, 8)}

    v = list(range(5))
    assert match([0, 1, x @ SequenceOf(...), 4], v, ctx := {}) == v
    assert ctx == {"x": (2, 3)}
    assert match([0, 1, "var" @ SequenceOf(...), 4], v, ctx := {}) == v
    assert ctx == {"var": (2, 3)}

    p = [
        0,
        1,
        "ints" @ SequenceOf(InstanceOf(int)),
        "floats" @ SequenceOf(InstanceOf(float)),
        6,
    ]
    v = [0, 1, 2, 3, 4.0, 5.0, 6]
    assert match(p, v, ctx := {}) == v
    assert ctx == {"ints": (2, 3), "floats": (4.0, 5.0)}


def test_matching_sequence_remaining():
    Seq = SequenceOf
    IsInt = InstanceOf(int)

    three = [1, 2, 3]
    four = [1, 2, 3, 4]
    five = [1, 2, 3, 4, 5]

    assert match([1, 2, 3, Seq(IsInt, at_least=1)], four) == four
    assert match([1, 2, 3, Seq(IsInt, at_least=1)], three) is NoMatch
    assert match([1, 2, 3, Seq(IsInt)], three) == three
    assert match([1, 2, 3, Seq(IsInt, at_most=1)], three) == three
    assert match([1, 2, 3, Seq(IsInt & Between(0, 10))], five) == five
    assert match([1, 2, 3, Seq(IsInt & Between(0, 4))], five) is NoMatch
    assert match([1, 2, 3, Seq(IsInt, at_least=2)], four) is NoMatch
    assert match([1, 2, 3, "res" @ Seq(IsInt, at_least=2)], five, ctx := {}) == five
    assert ctx == {"res": (4, 5)}


def test_matching_sequence_complicated():
    pattern = [
        1,
        "a" @ ListOf(InstanceOf(int) & Check(lambda x: x < 10)),
        4,
        "b" @ SequenceOf(...),
        8,
        9,
    ]
    expected = {
        "a": [2, 3],
        "b": (5, 6, 7),
    }
    assert match(pattern, range(1, 10), ctx := {}) == list(range(1, 10))
    assert ctx == expected

    pattern = [0, "pairs" @ PatternSequence([-1, -2]), 3]
    expected = {"pairs": [-1, -2]}
    assert match(pattern, [0, -1, -2, 3], ctx := {}) == [0, -1, -2, 3]
    assert ctx == expected

    pattern = [
        0,
        "first" @ PatternSequence([1, 2]),
        "second" @ PatternSequence([4, 5]),
        3,
    ]
    expected = {"first": [1, 2], "second": [4, 5]}
    assert match(pattern, [0, 1, 2, 4, 5, 3], ctx := {}) == [0, 1, 2, 4, 5, 3]
    assert ctx == expected

    pattern = [1, 2, "remaining" @ SequenceOf(...)]
    expected = {"remaining": (3, 4, 5, 6, 7, 8, 9)}
    assert match(pattern, range(1, 10), ctx := {}) == list(range(1, 10))
    assert ctx == expected

    assert match([0, SequenceOf([1, 2]), 3], v := [0, [1, 2], [1, 2], 3]) == v


def test_pattern_map():
    assert PatternMapping({}).match({}, context={}) == {}
    assert PatternMapping({}).match({1: 2}, context={}) is NoMatch


def test_matching_mapping():
    assert match({}, {}) == {}
    assert match({}, {1: 2}) is NoMatch

    assert match({1: 2}, {1: 2}) == {1: 2}
    assert match({1: 2}, {1: 3}) is NoMatch

    assert match({}, 3) is NoMatch
    ctx = {}
    assert match({"a": "capture" @ InstanceOf(int)}, {"a": 1}, ctx) == {"a": 1}
    assert ctx == {"capture": 1}

    p = {
        "a": "capture" @ InstanceOf(int),
        "b": InstanceOf(float),
        ...: InstanceOf(str),
    }
    ctx = {}
    assert match(p, {"a": 1, "b": 2.0, "c": "foo"}, ctx) == {
        "a": 1,
        "b": 2.0,
        "c": "foo",
    }
    assert ctx == {"capture": 1}
    assert match(p, {"a": 1, "b": 2.0, "c": 3}) is NoMatch

    p = {
        "a": "capture" @ InstanceOf(int),
        "b": InstanceOf(float),
        "rest" @ SequenceOf(...): InstanceOf(str),
    }
    ctx = {}
    assert match(p, {"a": 1, "b": 2.0, "c": "foo"}, ctx) == {
        "a": 1,
        "b": 2.0,
        "c": "foo",
    }
    assert ctx == {"capture": 1, "rest": ("c",)}


@pytest.mark.parametrize(
    ("pattern", "value", "expected"),
    [
        (InstanceOf(bool), True, True),
        (InstanceOf(str), "foo", "foo"),
        (InstanceOf(int), 8, 8),
        (InstanceOf(int), 1, 1),
        (InstanceOf(float), 1.0, 1.0),
        (IsIn({"a", "b"}), "a", "a"),
        (IsIn({"a": 1, "b": 2}), "a", "a"),
        (IsIn(["a", "b"]), "a", "a"),
        (IsIn(("a", "b")), "b", "b"),
        (IsIn({"a", "b", "c"}), "c", "c"),
        (TupleOf(InstanceOf(int)), (1, 2, 3), (1, 2, 3)),
        (TupleOf((InstanceOf(int), InstanceOf(str))), (1, "a"), (1, "a")),
        (ListOf(InstanceOf(str)), ["a", "b"], ["a", "b"]),
        (AnyOf(InstanceOf(str), InstanceOf(int)), "foo", "foo"),
        (AnyOf(InstanceOf(str), InstanceOf(int)), 7, 7),
        (
            AllOf(InstanceOf(int), Check(lambda v: v >= 3), Check(lambda v: v >= 8)),
            10,
            10,
        ),
        (
            MappingOf(InstanceOf(str), InstanceOf(int)),
            {"a": 1, "b": 2},
            {"a": 1, "b": 2},
        ),
    ],
)
def test_various_patterns(pattern, value, expected):
    assert pattern.match(value, context={}) == expected


@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (InstanceOf(bool), "foo"),
        (InstanceOf(str), True),
        (InstanceOf(int), 8.1),
        (Min(3), 2),
        (InstanceOf(int), None),
        (InstanceOf(float), 1),
        (IsIn(["a", "b"]), "c"),
        (IsIn({"a", "b"}), "c"),
        (IsIn({"a": 1, "b": 2}), "d"),
        (TupleOf(InstanceOf(int)), (1, 2.0, 3)),
        (ListOf(InstanceOf(str)), ["a", "b", None]),
        (AnyOf(InstanceOf(str), Min(4)), 3.14),
        (AnyOf(InstanceOf(str), Min(10)), 9),
        (AllOf(InstanceOf(int), Min(3), Min(8)), 7),
        (DictOf(InstanceOf(int), InstanceOf(str)), {"a": 1, "b": 2}),
    ],
)
def test_various_not_matching_patterns(pattern, value):
    assert pattern.match(value, context={}) is NoMatch


@pattern
def endswith_d(s, ctx):
    if not s.endswith("d"):
        return NoMatch
    return s


def test_pattern_decorator():
    assert endswith_d.match("abcd", context={}) == "abcd"
    assert endswith_d.match("abc", context={}) is NoMatch


@pytest.mark.parametrize(
    ("annot", "expected"),
    [
        (int, InstanceOf(int)),
        (str, InstanceOf(str)),
        (bool, InstanceOf(bool)),
        (Optional[int], Option(InstanceOf(int))),
        (Optional[Union[str, int]], Option(AnyOf(InstanceOf(str), InstanceOf(int)))),
        (Union[int, str], AnyOf(InstanceOf(int), InstanceOf(str))),
        (Annotated[int, Min(3)], AllOf(InstanceOf(int), Min(3))),
        (list[int], SequenceOf(InstanceOf(int), list)),
        (
            tuple[int, float, str],
            TupleOf((InstanceOf(int), InstanceOf(float), InstanceOf(str))),
        ),
        (tuple[int, ...], TupleOf(InstanceOf(int))),
        (
            dict[str, float],
            DictOf(InstanceOf(str), InstanceOf(float)),
        ),
        (FrozenDict[str, int], FrozenDictOf(InstanceOf(str), InstanceOf(int))),
        (Literal["alpha", "beta", "gamma"], IsIn(("alpha", "beta", "gamma"))),
        (
            Callable[[str, int], str],
            CallableWith((InstanceOf(str), InstanceOf(int)), InstanceOf(str)),
        ),
        (Callable, InstanceOf(CallableABC)),
    ],
)
def test_pattern_from_typehint(annot, expected):
    assert Pattern.from_typehint(annot) == expected


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_pattern_from_typehint_uniontype():
    # uniontype marks `type1 | type2` annotations and it's different from
    # Union[type1, type2]
    validator = Pattern.from_typehint(str | int | float)
    assert validator == AnyOf(InstanceOf(str), InstanceOf(int), InstanceOf(float))


def test_pattern_from_typehint_disable_coercion():
    class MyFloat(float, Coercible):
        @classmethod
        def __coerce__(cls, obj):
            return cls(float(obj))

    p = Pattern.from_typehint(MyFloat, allow_coercion=True)
    assert isinstance(p, CoercedTo)

    p = Pattern.from_typehint(MyFloat, allow_coercion=False)
    assert isinstance(p, InstanceOf)


class PlusOne(Coercible):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    @classmethod
    def __coerce__(cls, obj):
        return cls(obj + 1)

    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value


class PlusOneRaise(PlusOne):
    @classmethod
    def __coerce__(cls, obj):
        if isinstance(obj, cls):
            return obj
        else:
            raise ValueError("raise on coercion")


class PlusOneChild(PlusOne):
    pass


class PlusTwo(PlusOne):
    @classmethod
    def __coerce__(cls, obj):
        return obj + 2


def test_pattern_from_coercible_protocol():
    s = Pattern.from_typehint(PlusOne)
    assert s.match(1, context={}) == PlusOne(2)
    assert s.match(10, context={}) == PlusOne(11)


def test_pattern_coercible_bypass_coercion():
    s = Pattern.from_typehint(PlusOneRaise)
    # bypass coercion since it's already an instance of SomethingRaise
    assert s.match(PlusOneRaise(10), context={}) == PlusOneRaise(10)
    # but actually call __coerce__ if it's not an instance
    with pytest.raises(ValueError, match="raise on coercion"):
        s.match(10, context={})


def test_pattern_coercible_checks_type():
    s = Pattern.from_typehint(PlusOneChild)
    v = Pattern.from_typehint(PlusTwo)

    assert s.match(1, context={}) == PlusOneChild(2)

    assert PlusTwo.__coerce__(1) == 3
    assert v.match(1, context={}) is NoMatch


class DoubledList(Coercible, list[T]):
    @classmethod
    def __coerce__(cls, obj):
        return cls(list(obj) * 2)


def test_pattern_coercible_sequence_type():
    s = Pattern.from_typehint(Sequence[PlusOne])
    with pytest.raises(TypeError, match=r"Sequence\(\) takes no arguments"):
        s.match([1, 2, 3], context={})

    s = Pattern.from_typehint(list[PlusOne])
    assert s == SequenceOf(CoercedTo(PlusOne), type=list)
    assert s.match([1, 2, 3], context={}) == [PlusOne(2), PlusOne(3), PlusOne(4)]

    s = Pattern.from_typehint(tuple[PlusOne, ...])
    assert s == TupleOf(CoercedTo(PlusOne))
    assert s.match([1, 2, 3], context={}) == (PlusOne(2), PlusOne(3), PlusOne(4))

    s = Pattern.from_typehint(DoubledList[PlusOne])
    assert s == SequenceOf(CoercedTo(PlusOne), type=DoubledList)
    assert s.match([1, 2, 3], context={}) == DoubledList(
        [PlusOne(2), PlusOne(3), PlusOne(4), PlusOne(2), PlusOne(3), PlusOne(4)]
    )


def test_pattern_function():
    class MyNegativeInt(int, Coercible):
        @classmethod
        def __coerce__(cls, other):
            return cls(-int(other))

    class Box(Generic[T]):
        value: T

    def f(x):
        return x > 0

    # ... is treated the same as Any()
    assert pattern(...) == Any()
    assert pattern(Any()) == Any()
    assert pattern(True) == EqualTo(True)

    # plain types are converted to InstanceOf patterns
    assert pattern(int) == InstanceOf(int)
    # no matter whether the type implements the coercible protocol or not
    assert pattern(MyNegativeInt) == InstanceOf(MyNegativeInt)

    # generic types are converted to GenericInstanceOf patterns
    assert pattern(Box[int]) == GenericInstanceOf(Box[int])
    # no matter whethwe the origin type implements the coercible protocol or not
    assert pattern(Box[MyNegativeInt]) == GenericInstanceOf(Box[MyNegativeInt])

    # sequence typehints are converted to the appropriate sequence checkers
    assert pattern(List[int]) == ListOf(InstanceOf(int))  # noqa: UP006

    # spelled out sequences construct a more advanced pattern sequence
    assert pattern([int, str, 1]) == PatternSequence(
        [InstanceOf(int), InstanceOf(str), EqualTo(1)]
    )

    # matching deferred to user defined functions
    assert pattern(f) == Custom(f)


class Term(GraphNode):
    def __eq__(self, other):
        return type(self) is type(other) and self.__args__ == other.__args__

    def __hash__(self):
        return hash((self.__class__, self.__args__))

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(repr, self.__args__))})"


class Lit(Term):
    __slots__ = ("value",)
    __argnames__ = ("value",)
    __match_args__ = ("value",)

    def __init__(self, value):
        self.value = value

    @property
    def __args__(self):
        return (self.value,)


class Binary(Term):
    __slots__ = ("left", "right")
    __argnames__ = ("left", "right")
    __match_args__ = ("left", "right")

    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def __args__(self):
        return (self.left, self.right)


class Add(Binary):
    pass


class Mul(Binary):
    pass


one = Lit(1)
two = Mul(Lit(2), one)

three = Add(one, two)
six = Mul(two, three)
seven = Add(one, six)
fourteen = Add(seven, seven)


def test_node():
    pat = Node(
        InstanceOf(Add),
        each_arg=Replace(Object(Lit, value=Capture("v")), lambda _, v: Lit(v + 100)),
    )
    result = six.replace(pat)
    assert result == Mul(two, Add(Lit(101), two))
