from __future__ import annotations

import re
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import pytest
from typing_extensions import Annotated

from ibis.common.collections import frozendict
from ibis.common.patterns import (
    AllOf,
    Any,
    AnyOf,
    CallableWith,
    Capture,
    Check,
    CoercedTo,
    Contains,
    DictOf,
    EqualTo,
    FrozenDictOf,
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
    TupleOf,
    TypeOf,
    match,
)


class Double(Pattern):
    def match(self, value, *, context):
        return value * 2

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


def test_any():
    p = Any()
    assert p.match(1, context={}) == 1
    assert p.match("foo", context={}) == "foo"


def test_reference():
    p = Reference("other")
    context = {"other": 10}
    assert p.match(context=context) == 10


def test_capture():
    p = Capture(Double(), "result")
    assert p.match(10, context={}) == 20


def test_option():
    p = Option(InstanceOf(int), 1)
    assert p.match(11, context={}) == 11
    assert p.match(None, context={}) == 1
    assert p.match(None, context={}) == 1

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


def test_coerced_to():
    class MyInt(int):
        @classmethod
        def __coerce__(cls, other):
            return MyInt(other) + 1

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

    p = MappingOf(InstanceOf(str), InstanceOf(str), frozendict)
    assert p.match({"foo": "bar"}, context={}) == frozendict({"foo": "bar"})
    assert p.match({"foo": 1}, context={}) is NoMatch


def test_object_pattern():
    class Foo:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    assert match(Object(Foo, 1, b=2), Foo(1, 2)) == {}


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

    # wrapped = callable_with([instance_of(int)] * 4, instance_of(int), func_with_args)
    # assert wrapped(1, 2, 3, 4) == 10

    # wrapped = callable_with(
    #     [instance_of(int), instance_of(str)], instance_of(str), func
    # )
    # assert wrapped(1, "st") == "1st"

    # msg = "Given argument with type <class 'int'> is not an instance of <class 'str'>"
    # with pytest.raises(TypeError, match=msg):
    #     wrapped(1, 2)


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
    # assert match([1, 2, 3, Seq(IsInt(int) & max_(10))], [1, 2, 3, 4, 5]) == {}
    # assert match([1, 2, 3, Seq(IsInt(int) & max_(4))], [1, 2, 3, 4, 5]) is NoMatch
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


@pytest.mark.parametrize(
    ("annot", "expected"),
    [
        (int, InstanceOf(int)),
        (str, InstanceOf(str)),
        (bool, InstanceOf(bool)),
        (Optional[int], AnyOf(InstanceOf(int), InstanceOf(type(None)))),
        (Union[int, str], AnyOf(InstanceOf(int), InstanceOf(str))),
        (Annotated[int, Min(3)], AllOf(InstanceOf(int), Min(3))),
        # (
        #     Annotated[str, short_str, endswith_d],
        #     AllOf((InstanceOf(str), short_str, endswith_d)),
        # ),
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
        (frozendict[str, int], FrozenDictOf(InstanceOf(str), InstanceOf(int))),
        (Literal["alpha", "beta", "gamma"], IsIn(("alpha", "beta", "gamma"))),
        (
            Callable[[str, int], str],
            CallableWith((InstanceOf(str), InstanceOf(int)), InstanceOf(str)),
        ),
        (Callable, InstanceOf(Callable)),
    ],
)
def test_pattern_from_typehint(annot, expected):
    assert Pattern.from_typehint(annot) == expected
