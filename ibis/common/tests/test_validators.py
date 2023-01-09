from typing import Dict, List, Optional, Tuple, Union

import pytest
from typing_extensions import Annotated

from ibis.common.validators import (
    Validator,
    all_of,
    any_of,
    bool_,
    dict_of,
    frozendict_of,
    instance_of,
    int_,
    isin,
    list_of,
    min_,
    str_,
    tuple_of,
)
from ibis.util import frozendict


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
        (isin({"a": 1, "b": 2}), "a", 1),
        (tuple_of(instance_of(int)), (1, 2, 3), (1, 2, 3)),
        (list_of(instance_of(str)), ["a", "b"], ["a", "b"]),
        (any_of((str_, int_(max=8))), "foo", "foo"),
        (any_of((str_, int_(max=8))), 7, 7),
        (all_of((int_, min_(3), min_(8))), 10, 10),
        (dict_of(str_, int_), {"a": 1, "b": 2}, {"a": 1, "b": 2}),
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
        (isin({"a", "b"}), "c"),
        (isin({"a": 1, "b": 2}), "d"),
        (tuple_of(instance_of(int)), (1, 2.0, 3)),
        (list_of(instance_of(str)), ["a", "b", None]),
        (any_of((str_, int_(max=8))), 3.14),
        (any_of((str_, int_(max=8))), 9),
        (all_of((int_, min_(3), min_(8))), 7),
        (dict_of(int_, str_), {"a": 1, "b": 2}),
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
        (List[int], list_of(instance_of(int))),
        (Tuple[int], tuple_of(instance_of(int))),
        (Dict[str, float], dict_of(instance_of(str), instance_of(float))),
        (frozendict[str, int], frozendict_of(instance_of(str), instance_of(int))),
    ],
)
def test_validator_from_annotation(annot, expected):
    validator = Validator.from_annotation(annot)
    assert validator == expected
