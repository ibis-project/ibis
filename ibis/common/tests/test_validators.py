import pytest
from toolz import identity

from ibis.common.validators import Optional, Validator


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
    assert validator(None) == expected


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
    assert validator(value) == expected


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
