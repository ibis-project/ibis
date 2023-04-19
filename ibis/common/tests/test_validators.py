from __future__ import annotations

import sys
from typing import (
    Callable,
    Dict,
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

from ibis.common.collections import frozendict
from ibis.common.validators import (
    Coercible,
    Validator,
    all_of,
    any_of,
    bool_,
    callable_with,
    coerced_to,
    dict_of,
    equal_to,
    frozendict_of,
    instance_of,
    int_,
    isin,
    list_of,
    mapping_of,
    min_,
    pair_of,
    ref,
    sequence_of,
    str_,
    tuple_of,
)

T = TypeVar("T")


def test_ref():
    assert ref("b", this={"a": 1, "b": 2}) == 2


@pytest.mark.parametrize(
    ('validator', 'value', 'expected'),
    [
        (bool_, True, True),
        (str_, "foo", "foo"),
        (int_, 8, 8),
        (int_(min=10), 11, 11),
        (min_(3), 5, 5),
        (instance_of(int), 1, 1),
        (instance_of(float), 1.0, 1.0),
        (isin({"a", "b"}), "a", "a"),
        (isin({"a": 1, "b": 2}), "a", "a"),
        (isin(['a', 'b']), 'a', 'a'),
        (isin(('a', 'b')), 'b', 'b'),
        (isin({'a', 'b', 'c'}), 'c', 'c'),
        (tuple_of(instance_of(int)), (1, 2, 3), (1, 2, 3)),
        (tuple_of((instance_of(int), instance_of(str))), (1, "a"), (1, "a")),
        (list_of(instance_of(str)), ["a", "b"], ["a", "b"]),
        (any_of((str_, int_(max=8))), "foo", "foo"),
        (any_of((str_, int_(max=8))), 7, 7),
        (all_of((int_, min_(3), min_(8))), 10, 10),
        (dict_of(str_, int_), {"a": 1, "b": 2}, {"a": 1, "b": 2}),
        (pair_of(bool_, str_), (True, "foo"), (True, "foo")),
        (equal_to(1), 1, 1),
        (equal_to(None), None, None),
        (coerced_to(int), "1", 1),
    ],
)
def test_validators_passing(validator, value, expected):
    assert validator(value) == expected


@pytest.mark.parametrize(
    ('validator', 'value'),
    [
        (bool_, "foo"),
        (str_, True),
        (int_, 8.1),
        (int_(min=10), 9),
        (min_(3), 2),
        (instance_of(int), None),
        (instance_of(float), 1),
        (isin(["a", "b"]), "c"),
        (isin({"a", "b"}), "c"),
        (isin({"a": 1, "b": 2}), "d"),
        (tuple_of(instance_of(int)), (1, 2.0, 3)),
        (list_of(instance_of(str)), ["a", "b", None]),
        (any_of((str_, int_(max=8))), 3.14),
        (any_of((str_, int_(max=8))), 9),
        (all_of((int_, min_(3), min_(8))), 7),
        (dict_of(int_, str_), {"a": 1, "b": 2}),
        (pair_of(bool_, str_), (True, True, True)),
        (pair_of(bool_, str_), ("str", True)),
        (equal_to(1), 2),
    ],
)
def test_validators_failing(validator, value):
    with pytest.raises((TypeError, ValueError)):
        validator(value)


def short_str(x, this):
    return len(x) > 3


def endswith_d(x, this):
    return x.endswith("d")


@pytest.mark.parametrize(
    ("annot", "expected"),
    [
        (int, instance_of(int)),
        (str, instance_of(str)),
        (bool, instance_of(bool)),
        (Optional[int], any_of((instance_of(int), instance_of(type(None))))),
        (Union[int, str], any_of((instance_of(int), instance_of(str)))),
        (Annotated[int, min_(3)], all_of((instance_of(int), min_(3)))),
        (
            Annotated[str, short_str, endswith_d],
            all_of((instance_of(str), short_str, endswith_d)),
        ),
        (List[int], sequence_of(instance_of(int), type=coerced_to(list))),
        (
            Tuple[int, float, str],
            tuple_of(
                (instance_of(int), instance_of(float), instance_of(str)),
                type=coerced_to(tuple),
            ),
        ),
        (Tuple[int, ...], tuple_of(instance_of(int), type=coerced_to(tuple))),
        (
            Dict[str, float],
            dict_of(instance_of(str), instance_of(float), type=coerced_to(dict)),
        ),
        (
            frozendict[str, int],
            frozendict_of(
                instance_of(str), instance_of(int), type=coerced_to(frozendict)
            ),
        ),
        (Literal["alpha", "beta", "gamma"], isin(("alpha", "beta", "gamma"))),
        (
            Callable[[str, int], str],
            callable_with((instance_of(str), instance_of(int)), instance_of(str)),
        ),
        (Callable, instance_of(Callable)),
    ],
)
def test_validator_from_typehint(annot, expected):
    assert Validator.from_typehint(annot) == expected


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_validator_from_typehint_uniontype():
    # uniontype marks `type1 | type2` annotations and it's different from
    # Union[type1, type2]
    validator = Validator.from_typehint(str | int | float)
    assert validator == any_of((instance_of(str), instance_of(int), instance_of(float)))


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
        raise TypeError("raise on coercion")


class PlusOneChild(PlusOne):
    pass


class PlusTwo(PlusOne):
    @classmethod
    def __coerce__(cls, obj):
        return obj + 2


def test_coercible_protocol():
    s = Validator.from_typehint(PlusOne)
    assert s(1) == PlusOne(2)
    assert s(10) == PlusOne(11)


def test_coercible_bypass_coercion():
    s = Validator.from_typehint(PlusOneRaise)
    # bypass coercion since it's already an instance of SomethingRaise
    assert s(PlusOneRaise(10)) == PlusOneRaise(10)
    # but actually call __coerce__ if it's not an instance
    with pytest.raises(TypeError, match="raise on coercion"):
        s(10)


def test_coercible_checks_type():
    s = Validator.from_typehint(PlusOneChild)
    v = Validator.from_typehint(PlusTwo)

    assert s(1) == PlusOneChild(2)

    assert PlusTwo.__coerce__(1) == 3
    with pytest.raises(TypeError, match="not an instance of .*PlusTwo.*"):
        v(1)


class DoubledList(List[T]):
    @classmethod
    def __coerce__(cls, obj):
        return cls(list(obj) * 2)


def test_coercible_sequence_type():
    s = Validator.from_typehint(Sequence[PlusOne])
    with pytest.raises(TypeError, match=r"Sequence\(\) takes no arguments"):
        s([1, 2, 3])

    s = Validator.from_typehint(List[PlusOne])
    assert s == sequence_of(coerced_to(PlusOne), type=coerced_to(list))
    assert s([1, 2, 3]) == [PlusOne(2), PlusOne(3), PlusOne(4)]

    s = Validator.from_typehint(Tuple[PlusOne, ...])
    assert s == tuple_of(coerced_to(PlusOne), type=coerced_to(tuple))
    assert s([1, 2, 3]) == (PlusOne(2), PlusOne(3), PlusOne(4))

    s = Validator.from_typehint(DoubledList[PlusOne])
    assert s == sequence_of(coerced_to(PlusOne), type=coerced_to(DoubledList))
    assert s([1, 2, 3]) == DoubledList(
        [PlusOne(2), PlusOne(3), PlusOne(4), PlusOne(2), PlusOne(3), PlusOne(4)]
    )


def test_mapping_of():
    value = {"a": 1, "b": 2}
    assert mapping_of(str, int, value, type=dict) == value
    assert mapping_of(str, int, value, type=frozendict) == frozendict(value)

    with pytest.raises(TypeError, match="Argument must be a mapping"):
        mapping_of(str, float, 10, type=dict)


def test_callable_with():
    def func(a, b):
        return str(a) + b

    def func_with_args(a, b, *args):
        return sum((a, b) + args)

    def func_with_kwargs(a, b, c=1, **kwargs):
        return str(a) + b + str(c)

    def func_with_mandatory_kwargs(*, c):
        return c

    msg = "Argument must be a callable"
    with pytest.raises(TypeError, match=msg):
        callable_with([instance_of(int), instance_of(str)], 10, "string")

    msg = "Callable has mandatory keyword-only arguments which cannot be specified"
    with pytest.raises(TypeError, match=msg):
        callable_with([instance_of(int)], instance_of(str), func_with_mandatory_kwargs)

    msg = "Callable has more positional arguments than expected"
    with pytest.raises(TypeError, match=msg):
        callable_with([instance_of(int)] * 2, instance_of(str), func_with_kwargs)

    msg = "Callable has less positional arguments than expected"
    with pytest.raises(TypeError, match=msg):
        callable_with([instance_of(int)] * 4, instance_of(str), func_with_kwargs)

    wrapped = callable_with([instance_of(int)] * 4, instance_of(int), func_with_args)
    assert wrapped(1, 2, 3, 4) == 10

    wrapped = callable_with(
        [instance_of(int), instance_of(str)], instance_of(str), func
    )
    assert wrapped(1, "st") == "1st"

    msg = "Given argument with type <class 'int'> is not an instance of <class 'str'>"
    with pytest.raises(TypeError, match=msg):
        wrapped(1, 2)
