import inspect

import pytest
from toolz import identity

from ibis.common.validators import Optional, Parameter, Signature, Validator


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


def test_parameter():
    def fn(x, this):
        return int(x) + this['other']

    p = Parameter('novalidator')
    assert p.validate('value', this={}) == 'value'

    p = Parameter('test', validator=fn)

    assert p.validator is fn
    assert p.default is inspect.Parameter.empty
    assert p.validate('2', this={'other': 1}) == 3

    with pytest.raises(TypeError):
        p.validate({}, valid=inspect.Parameter.empty)

    ofn = Optional(fn)
    op = Parameter('test', validator=ofn)
    assert op.validator is ofn
    assert op.default is None
    assert op.validate(None, this={'other': 1}) is None


def test_signature():
    def to_int(x, this):
        return int(x)

    def add_other(x, this):
        return int(x) + this['other']

    other = Parameter('other', validator=to_int)
    this = Parameter('this', validator=add_other)

    sig = Signature(parameters=[other, this])
    assert sig.validate(1, 2) == {'other': 1, 'this': 3}
    assert sig.validate(other=1, this=2) == {'other': 1, 'this': 3}
    assert sig.validate(this=2, other=1) == {'other': 1, 'this': 3}
