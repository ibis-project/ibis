import pytest
from toolz import identity

from ibis.common.exceptions import IbisTypeError
from ibis.common.validators import Optional, Validator, instance_of, isin


class InstanceOf(Validator):
    def __init__(self, typ):
        self.typ = typ

    def __call__(self, arg, **kwargs):
        if not isinstance(arg, self.typ):
            raise TypeError(self.typ)
        return arg


IsInt = InstanceOf(int)


@pytest.mark.parametrize(
    ('default', 'expected'),
    [(None, None), (0, 0), ('default', 'default'), (lambda: 3, 3)],
)
def test_optional_argument(default, expected):
    validator = Optional(lambda x: x, default=default)
    assert validator.validate(None) == expected


@pytest.mark.parametrize(
    ('validator', 'value', 'expected'),
    [
        (Optional(identity, default=None), None, None),
        (Optional(identity, default=None), 'three', 'three'),
        (Optional(identity, default=1), None, 1),
        (Optional(identity, default=lambda: 8), 'cat', 'cat'),
        (Optional(identity, default=lambda: 8), None, 8),
        (Optional(int, default=11), None, 11),
        (Optional(int, default=None), None, None),
        (Optional(int, default=None), 18, 18),
        (Optional(str, default=None), 'caracal', 'caracal'),
    ],
)
def test_valid_optional(validator, value, expected):
    assert validator.validate(value) == expected


@pytest.mark.parametrize(
    ('arg', 'value', 'expected'),
    [
        (Optional(IsInt, default=''), None, TypeError),
        (Optional(IsInt), 'lynx', TypeError),
    ],
)
def test_invalid_optional(arg, value, expected):
    with pytest.raises(expected):
        arg(value)


@pytest.mark.parametrize(
    ('klass', 'value', 'expected'),
    [(int, 32, 32), (str, 'foo', 'foo'), (bool, True, True)],
)
def test_valid_instance_of(klass, value, expected):
    assert instance_of(klass).validate(value) == expected


@pytest.mark.parametrize(
    ('klass', 'value', 'expected'),
    [
        (Validator, object, IbisTypeError),
        (str, 4, IbisTypeError),
    ],
)
def test_invalid_instance_of(klass, value, expected):
    with pytest.raises(expected):
        assert instance_of(klass).validate(value)


@pytest.mark.parametrize(
    ('values', 'value', 'expected'),
    [
        (['a', 'b'], 'a', 'a'),
        (('a', 'b'), 'b', 'b'),
        ({'a', 'b', 'c'}, 'c', 'c'),
        ([1, 2, 'f'], 'f', 'f'),
        ({'a': 1, 'b': 2}, 'a', 1),
        ({'a': 1, 'b': 2}, 'b', 2),
    ],
)
def test_valid_isin(values, value, expected):
    assert isin(values).validate(value) == expected


@pytest.mark.parametrize(
    ('values', 'value', 'expected'),
    [
        (['a', 'b'], 'c', ValueError),
        ({'a', 'b', 'c'}, 'd', ValueError),
        ({'a': 1, 'b': 2}, 'c', ValueError),
    ],
)
def test_invalid_isin(values, value, expected):
    with pytest.raises(expected):
        isin(values).validate(value)
