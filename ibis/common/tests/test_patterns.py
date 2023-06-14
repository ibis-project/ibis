from __future__ import annotations

import re
import sys
from collections.abc import Callable as CallableABC
from dataclasses import dataclass
from typing import (
    Any as AnyType,
)
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import pytest
from typing_extensions import Annotated

from ibis.common.collections import FrozenDict
from ibis.common.graph import Node
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
    DictOf,
    EqualTo,
    FrozenDictOf,
    Function,
    Innermost,
    InstanceOf,
    IsIn,
    LazyInstanceOf,
    Length,
    ListOf,
    MappingOf,
    MatchError,
    NoMatch,
    NoneOf,
    Not,
    Object,
    Option,
    Pattern,
    PatternMapping,
    PatternSequence,
    Reference,
    SequenceOf,
    SubclassOf,
    Topmost,
    TupleOf,
    TypeOf,
    ValidationError,
    match,
    pattern,
)


class Double(Pattern):
    def match(self, value, *, context):
        return value * 2

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


class Min(Pattern):
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


def test_reference():
    p = Reference("other")
    context = {"other": 10}
    assert p.match(context=context) == 10


def test_capture():
    ctx = {}

    p = Capture(Min(11), "result")
    assert p.match(10, context=ctx) is NoMatch
    assert ctx == {}

    assert p.match(12, context=ctx) == 12
    assert ctx == {"result": 12}


def test_option():
    p = Option(InstanceOf(str))
    assert p.match(None, context={}) is None
    assert p.match("foo", context={}) == "foo"
    assert p.match(1, context={}) is NoMatch


def test_check():
    p = Check(lambda x: x == 10)
    assert p.match(10, context={}) == 10
    assert p.match(11, context={}) is NoMatch


def test_equal_to():
    p = EqualTo(10)
    assert p.match(10, context={}) == 10
    assert p.match(11, context={}) is NoMatch


def test_type_of():
    p = TypeOf(int)
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) is NoMatch


def test_subclass_of():
    p = SubclassOf(Pattern)
    assert p.match(Double, context={}) == Double
    assert p.match(int, context={}) is NoMatch


def test_instance_of():
    p = InstanceOf(int)
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) is NoMatch


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

    assert match(My[int, AnyType], My(1, 2, "3"), context={}) == {}
    assert match(My[int, int], My(1, 2, "3"), context={}) == {}
    assert match(My[int, float], My(1, 2, "3"), context={}) is NoMatch
    assert match(My[int, float], My(1, 2.0, "3"), context={}) == {}


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

        def output_dtype(self) -> T:
            ...

        def output_shape(self) -> S:
            ...

    class Literal(Value[T, Scalar]):
        def __init__(self, value, dtype):
            self.value = value
            self.dtype = dtype

        def output_dtype(self) -> T:
            return self.dtype

        def output_shape(self) -> DataShape:
            return Scalar()

        def __eq__(self, other):
            return (
                type(self) == type(other)
                and self.value == other.value
                and self.dtype == other.dtype
            )

    p = Pattern.from_typehint(Literal[String])
    r = p.match("foo", context={})
    assert r == Literal("foo", Scalar())


def test_not():
    p = Not(InstanceOf(int))
    p1 = ~InstanceOf(int)

    assert p == p1
    assert p.match(1, context={}) is NoMatch
    assert p.match("foo", context={}) == "foo"


def test_any_of():
    p = AnyOf(InstanceOf(int), InstanceOf(str))
    p1 = InstanceOf(int) | InstanceOf(str)

    assert p == p1
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) == "foo"
    assert p.match(1.0, context={}) is NoMatch


def test_all_of():
    def negative(x):
        return x < 0

    p = AllOf(InstanceOf(int), Check(negative))
    p1 = InstanceOf(int) & Check(negative)

    assert p == p1
    assert p.match(1, context={}) is NoMatch
    assert p.match(-1, context={}) == -1


def test_none_of():
    def negative(x):
        return x < 0

    p = NoneOf(InstanceOf(int), Check(negative))
    assert p.match(1.0, context={}) == 1.0
    assert p.match(-1.0, context={}) is NoMatch
    assert p.match(1, context={}) is NoMatch


def test_length():
    with pytest.raises(ValueError):
        Length(exactly=3, at_least=3)
    with pytest.raises(ValueError):
        Length(exactly=3, at_most=3)

    p = Length(exactly=3)
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([1, 2], context={}) is NoMatch

    p = Length(at_least=3)
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([1, 2], context={}) is NoMatch

    p = Length(at_most=3)
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([1, 2, 3, 4], context={}) is NoMatch

    p = Length(at_least=3, at_most=5)
    assert p.match([1, 2], context={}) is NoMatch
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([1, 2, 3, 4], context={}) == [1, 2, 3, 4]
    assert p.match([1, 2, 3, 4, 5], context={}) == [1, 2, 3, 4, 5]
    assert p.match([1, 2, 3, 4, 5, 6], context={}) is NoMatch


def test_contains():
    p = Contains(1)
    assert p.match([1, 2, 3], context={}) == [1, 2, 3]
    assert p.match([2, 3], context={}) is NoMatch


def test_isin():
    p = IsIn([1, 2, 3])
    assert p.match(1, context={}) == 1
    assert p.match(4, context={}) is NoMatch


def test_sequence_of():
    p = SequenceOf(InstanceOf(str), list)
    assert p.match(["foo", "bar"], context={}) == ["foo", "bar"]
    assert p.match([1, 2], context={}) is NoMatch
    assert p.match(1, context={}) is NoMatch


def test_list_of():
    p = ListOf(InstanceOf(str))
    assert p.match(["foo", "bar"], context={}) == ["foo", "bar"]
    assert p.match([1, 2], context={}) is NoMatch
    assert p.match(1, context={}) is NoMatch


def test_tuple_of():
    p = TupleOf((InstanceOf(str), InstanceOf(int), InstanceOf(float)))
    assert p.match(("foo", 1, 1.0), context={}) == ("foo", 1, 1.0)
    assert p.match(["foo", 1, 1.0], context={}) == ("foo", 1, 1.0)
    assert p.match(1, context={}) is NoMatch

    p = TupleOf(InstanceOf(str))
    assert p == SequenceOf(InstanceOf(str), tuple)
    assert p.match(("foo", "bar"), context={}) == ("foo", "bar")
    assert p.match(["foo"], context={}) == ("foo",)
    assert p.match(1, context={}) is NoMatch


def test_mapping_of():
    p = MappingOf(InstanceOf(str), InstanceOf(int))
    assert p.match({"foo": 1, "bar": 2}, context={}) == {"foo": 1, "bar": 2}
    assert p.match({"foo": 1, "bar": "baz"}, context={}) is NoMatch
    assert p.match(1, context={}) is NoMatch

    p = MappingOf(InstanceOf(str), InstanceOf(str), FrozenDict)
    assert p.match({"foo": "bar"}, context={}) == FrozenDict({"foo": "bar"})
    assert p.match({"foo": 1}, context={}) is NoMatch


def test_object_pattern():
    class Foo:
        __match_args__ = ("a", "b")

        def __init__(self, a, b):
            self.a = a
            self.b = b

    p = Object(Foo, 1, b=2)
    assert match(p, Foo(1, 2)) == {}


def test_callable_with():
    def func(a, b):
        return str(a) + b

    def func_with_args(a, b, *args):
        return sum((a, b) + args)

    def func_with_kwargs(a, b, c=1, **kwargs):
        return str(a) + b + str(c)

    def func_with_mandatory_kwargs(*, c):
        return c

    p = CallableWith([InstanceOf(int), InstanceOf(str)])
    assert p.match(10, context={}) is NoMatch

    msg = "Callable has mandatory keyword-only arguments which cannot be specified"
    with pytest.raises(MatchError, match=msg):
        p.match(func_with_mandatory_kwargs, context={})

    # Callable has more positional arguments than expected
    p = CallableWith([InstanceOf(int)] * 2)
    assert p.match(func_with_kwargs, context={}) is NoMatch

    # Callable has less positional arguments than expected
    p = CallableWith([InstanceOf(int)] * 4)
    assert p.match(func_with_kwargs, context={}) is NoMatch

    p = CallableWith([InstanceOf(int)] * 4, InstanceOf(int))
    wrapped = p.match(func_with_args, context={})
    assert wrapped(1, 2, 3, 4) == 10

    p = CallableWith([InstanceOf(int), InstanceOf(str)], InstanceOf(str))
    wrapped = p.match(func, context={})
    assert wrapped(1, "st") == "1st"

    with pytest.raises(ValidationError, match="2 doesn't match InstanceOf"):
        wrapped(1, 2)


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
    assert match("foo", "foo") == {}
    assert match("foo", "bar") is NoMatch

    assert match(InstanceOf(int), 1) == {}
    assert match(InstanceOf(int), "foo") is NoMatch

    assert Capture(InstanceOf(float), "pi") == "pi" @ InstanceOf(float)
    assert Capture(InstanceOf(float), "pi") == InstanceOf(float) >> "pi"

    assert match(Capture(InstanceOf(float), "pi"), 3.14) == {"pi": 3.14}
    assert match("pi" @ InstanceOf(float), 3.14) == {"pi": 3.14}

    assert match(InstanceOf(int) | InstanceOf(float), 3) == {}
    assert match(InstanceOf(object) & InstanceOf(float), 3.14) == {}


def test_matching_sequence_pattern():
    assert match([], []) == {}
    assert match([], [1]) is NoMatch

    assert match([1, 2, 3, 4, ...], list(range(1, 9))) == {}
    assert match([1, 2, 3, 4, ...], list(range(1, 3))) is NoMatch
    assert match([1, 2, 3, 4, ...], list(range(1, 5))) == {}
    assert match([1, 2, 3, 4, ...], list(range(1, 6))) == {}

    assert match([..., 3, 4], list(range(5))) == {}
    assert match([..., 3, 4], list(range(3))) is NoMatch

    assert match([0, 1, ..., 4], list(range(5))) == {}
    assert match([0, 1, ..., 4], list(range(4))) is NoMatch

    assert match([...], list(range(5))) == {}
    assert match([..., 2, 3, 4, ...], list(range(8))) == {}


def test_matching_sequence_with_captures():
    assert match([1, 2, 3, 4, SequenceOf(...)], list(range(1, 9))) == {}
    assert match([1, 2, 3, 4, "rest" @ SequenceOf(...)], list(range(1, 9))) == {
        "rest": (5, 6, 7, 8)
    }

    assert match([0, 1, "var" @ SequenceOf(...), 4], list(range(5))) == {"var": (2, 3)}
    assert match([0, 1, SequenceOf(...) >> "var", 4], list(range(5))) == {"var": (2, 3)}

    p = [
        0,
        1,
        "ints" @ SequenceOf(InstanceOf(int)),
        "floats" @ SequenceOf(InstanceOf(float)),
        6,
    ]
    assert match(p, [0, 1, 2, 3, 4.0, 5.0, 6]) == {"ints": (2, 3), "floats": (4.0, 5.0)}


def test_matching_sequence_remaining():
    Seq = SequenceOf
    IsInt = InstanceOf(int)

    assert match([1, 2, 3, Seq(IsInt, at_least=1)], [1, 2, 3, 4]) == {}
    assert match([1, 2, 3, Seq(IsInt, at_least=1)], [1, 2, 3]) is NoMatch
    assert match([1, 2, 3, Seq(IsInt)], [1, 2, 3]) == {}
    assert match([1, 2, 3, Seq(IsInt, at_most=1)], [1, 2, 3]) == {}
    assert match([1, 2, 3, Seq(IsInt & Between(0, 10))], [1, 2, 3, 4, 5]) == {}
    assert match([1, 2, 3, Seq(IsInt & Between(0, 4))], [1, 2, 3, 4, 5]) is NoMatch
    assert match([1, 2, 3, Seq(IsInt, at_least=2)], [1, 2, 3, 4]) is NoMatch
    assert match([1, 2, 3, Seq(IsInt, at_least=2) >> "res"], [1, 2, 3, 4, 5]) == {
        "res": (4, 5)
    }


def test_matching_sequence_complicated():
    pattern = [
        1,
        'a' @ ListOf(InstanceOf(int) & Check(lambda x: x < 10)),
        4,
        'b' @ SequenceOf(...),
        8,
        9,
    ]
    expected = {
        "a": [2, 3],
        "b": (5, 6, 7),
    }
    assert match(pattern, range(1, 10)) == expected

    pattern = [0, PatternSequence([1, 2]) >> "pairs", 3]
    expected = {"pairs": [1, 2]}
    assert match(pattern, [0, 1, 2, 1, 2, 3]) == expected

    pattern = [
        0,
        PatternSequence([1, 2]) >> "first",
        PatternSequence([4, 5]) >> "second",
        3,
    ]
    expected = {"first": [1, 2], "second": [4, 5]}
    assert match(pattern, [0, 1, 2, 4, 5, 3]) == expected

    pattern = [1, 2, 'remaining' @ SequenceOf(...)]
    expected = {'remaining': (3, 4, 5, 6, 7, 8, 9)}
    assert match(pattern, range(1, 10)) == expected

    assert match([0, SequenceOf([1, 2]), 3], [0, [1, 2], [1, 2], 3]) == {}


def test_pattern_map():
    assert PatternMapping({}).match({}, context={}) == {}
    assert PatternMapping({}).match({1: 2}, context={}) is NoMatch


def test_matching_mapping():
    assert match({}, {}) == {}
    assert match({}, {1: 2}) is NoMatch

    assert match({1: 2}, {1: 2}) == {}
    assert match({1: 2}, {1: 3}) is NoMatch

    assert match({}, 3) is NoMatch
    assert match({'a': "capture" @ InstanceOf(int)}, {'a': 1}) == {"capture": 1}

    p = {
        "a": "capture" @ InstanceOf(int),
        "b": InstanceOf(float),
        ...: InstanceOf(str),
    }
    assert match(p, {"a": 1, "b": 2.0, "c": "foo"}) == {"capture": 1}
    assert match(p, {"a": 1, "b": 2.0, "c": 3}) is NoMatch

    p = {
        "a": "capture" @ InstanceOf(int),
        "b": InstanceOf(float),
        "rest" @ SequenceOf(...): InstanceOf(str),
    }
    assert match(p, {"a": 1, "b": 2.0, "c": "foo"}) == {"capture": 1, "rest": ("c",)}


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
        (IsIn(['a', 'b']), 'a', 'a'),
        (IsIn(('a', 'b')), 'b', 'b'),
        (IsIn({'a', 'b', 'c'}), 'c', 'c'),
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
    assert pattern.validate(value, context={}) == expected


@pytest.mark.parametrize(
    ('pattern', 'value'),
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
    with pytest.raises(ValidationError):
        pattern.validate(value, context={})


@pattern
def endswith_d(s, context):
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
        (List[int], SequenceOf(InstanceOf(int), list)),
        (
            Tuple[int, float, str],
            TupleOf((InstanceOf(int), InstanceOf(float), InstanceOf(str))),
        ),
        (Tuple[int, ...], TupleOf(InstanceOf(int))),
        (
            Dict[str, float],
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


class PlusOne(Coercible):
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
            raise TypeError("raise on coercion")


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
    with pytest.raises(TypeError, match="raise on coercion"):
        s.match(10, context={})


def test_pattern_coercible_checks_type():
    s = Pattern.from_typehint(PlusOneChild)
    v = Pattern.from_typehint(PlusTwo)

    assert s.match(1, context={}) == PlusOneChild(2)

    assert PlusTwo.__coerce__(1) == 3
    assert v.match(1, context={}) is NoMatch


class DoubledList(Coercible, List[T]):
    @classmethod
    def __coerce__(cls, obj):
        return cls(list(obj) * 2)


def test_pattern_coercible_sequence_type():
    s = Pattern.from_typehint(Sequence[PlusOne])
    with pytest.raises(TypeError, match=r"Sequence\(\) takes no arguments"):
        s.match([1, 2, 3], context={})

    s = Pattern.from_typehint(List[PlusOne])
    assert s == SequenceOf(CoercedTo(PlusOne), type=list)
    assert s.match([1, 2, 3], context={}) == [PlusOne(2), PlusOne(3), PlusOne(4)]

    s = Pattern.from_typehint(Tuple[PlusOne, ...])
    assert s == TupleOf(CoercedTo(PlusOne))
    assert s.match([1, 2, 3], context={}) == (PlusOne(2), PlusOne(3), PlusOne(4))

    s = Pattern.from_typehint(DoubledList[PlusOne])
    assert s == SequenceOf(CoercedTo(PlusOne), type=DoubledList)
    assert s.match([1, 2, 3], context={}) == DoubledList(
        [PlusOne(2), PlusOne(3), PlusOne(4), PlusOne(2), PlusOne(3), PlusOne(4)]
    )


def test_pattern_function():
    assert pattern(...) == Any()
    assert pattern(Any()) == Any()
    assert pattern(int) == InstanceOf(int)
    assert pattern(List[int]) == ListOf(InstanceOf(int))
    assert pattern([int, str, 1]) == PatternSequence(
        [InstanceOf(int), InstanceOf(str), EqualTo(1)]
    )
    assert pattern(True) == EqualTo(True)

    def f(x):
        return x > 0

    assert pattern(f) == Function(f)


class Term(Node):
    def __eq__(self, other):
        return type(self) is type(other) and self.__args__ == other.__args__

    def __hash__(self):
        return hash((self.__class__, self.__args__))


class Lit(Term):
    __argnames__ = ("value",)
    __match_args__ = ("value",)

    def __init__(self, value):
        self.value = value

    @property
    def __args__(self):
        return (self.value,)


class Binary(Term):
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


def test_topmost_innermost():
    inner = Object(Mul, Capture(Any(), "a"), Capture(Any(), "b"))
    assert inner.match(six, {}) is six

    context = {}
    p = Topmost(inner)
    m = p.match(seven, context)
    assert m is six
    assert context == {"a": two, "b": three}

    p = Innermost(inner)
    m = p.match(seven, context)
    assert m is two
    assert context == {"a": Lit(2), "b": one}
