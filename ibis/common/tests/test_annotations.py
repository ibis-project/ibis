import inspect

import pytest
from toolz import identity

from ibis.common.annotations import (
    Argument,
    Attribute,
    Default,
    Initialized,
    Mandatory,
    Optional,
    Parameter,
    Signature,
    Variadic,
)
from ibis.common.validators import Validator


class InstanceOf(Validator):
    def __init__(self, typ):
        self.typ = typ

    def __call__(self, arg, **kwargs):
        if not isinstance(arg, self.typ):
            raise TypeError(self.typ)
        return arg


IsInt = InstanceOf(int)


def test_default_argument():
    annotation = Default(validator=int, default=3)
    assert annotation.validate(1) == 1
    with pytest.raises(TypeError):
        annotation.validate(None)


@pytest.mark.parametrize(
    ('default', 'expected'),
    [(None, None), (0, 0), ('default', 'default'), (lambda: 3, 3)],
)
def test_optional_argument(default, expected):
    annotation = Optional(default=default)
    assert annotation.validate(None) == expected


@pytest.mark.parametrize(
    ('argument', 'value', 'expected'),
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
def test_valid_optional(argument, value, expected):
    assert argument.validate(value) == expected


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


def test_initialized():
    class Foo:
        a = 10

    field = Initialized(lambda self: self.a)
    assert field == field
    assert field.initialize(Foo) == 10

    field2 = Initialized(lambda self: self.a, validator=lambda x, this: str(x))
    assert field != field2
    assert field2.initialize(Foo) == '10'


def test_parameter():
    def fn(x, this):
        return int(x) + this['other']

    annot = Argument(fn)
    p = Parameter('test', annotation=annot)

    assert p.annotation is annot
    assert p.default is inspect.Parameter.empty
    assert p.validate('2', this={'other': 1}) == 3

    with pytest.raises(TypeError):
        p.validate({}, valid=inspect.Parameter.empty)

    ofn = Optional(fn)
    op = Parameter('test', annotation=ofn)
    assert op.annotation is ofn
    assert op.default is None
    assert op.validate(None, this={'other': 1}) is None

    with pytest.raises(TypeError, match="Invalid annotation type"):
        Parameter("wrong", annotation=Attribute("a"))


def test_signature():
    def to_int(x, this):
        return int(x)

    def add_other(x, this):
        return int(x) + this['other']

    other = Parameter('other', annotation=Mandatory(to_int))
    this = Parameter('this', annotation=Mandatory(add_other))

    sig = Signature(parameters=[other, this])
    assert sig.validate(1, 2) == {'other': 1, 'this': 3}
    assert sig.validate(other=1, this=2) == {'other': 1, 'this': 3}
    assert sig.validate(this=2, other=1) == {'other': 1, 'this': 3}


def test_signature_unbind():
    def to_int(x, this):
        return int(x)

    def add_other(x, this):
        return int(x) + this['other']

    other = Parameter('other', annotation=Mandatory(to_int))
    this = Parameter('this', annotation=Mandatory(add_other))

    sig = Signature(parameters=[other, this])
    params = sig.validate(1, this=2)

    args, kwargs = sig.unbind(params)
    assert args == ()
    assert kwargs == {"other": 1, "this": 3}


def as_float(x, this):
    return float(x)


a = Parameter('a', annotation=Mandatory(as_float))
b = Parameter('b', annotation=Mandatory(as_float))
c = Parameter('c', annotation=Default(as_float))
d = Parameter('d', annotation=Variadic(as_float))
e = Parameter('e', annotation=Mandatory(as_float), keyword=True)
sig = Signature(parameters=[a, b, c, d, e])


def test_signature_unbind_with_empty_variadic():
    params = sig.validate(1, 2, 3, e=4)
    assert params == {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': (), 'e': 4.0}

    args, kwargs = sig.unbind(params)
    assert args == ()
    assert kwargs == {'a': 1.0, 'b': 2.0, 'c': 3.0, 'e': 4.0}
    params_again = sig.validate(*args, **kwargs)
    assert params_again == params


def test_signature_unbind_nonempty_variadic():
    params = sig.validate(1, 2, 3, 10, 11, 12, 13, 14, e=4)
    assert params == {
        'a': 1.0,
        'b': 2.0,
        'c': 3.0,
        'd': (10, 11, 12, 13, 14),
        'e': 4.0,
    }

    args, kwargs = sig.unbind(params)
    assert args == (10, 11, 12, 13, 14)
    assert kwargs == {'a': 1.0, 'b': 2.0, 'c': 3.0, 'e': 4.0}
    # note that in this case the bind-unbind-bind roundtrip is not possible
    # but the returned args and kwargs can be used to call a user function
    # with the expected arguments just with different order
