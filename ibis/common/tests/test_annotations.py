from __future__ import annotations

import inspect
from typing import Union

import pytest
from typing_extensions import Annotated  # noqa: TCH002

from ibis.common.annotations import Argument, Attribute, Parameter, Signature, annotated
from ibis.common.patterns import (
    Any,
    CoercedTo,
    InstanceOf,
    NoMatch,
    Option,
    TupleOf,
    ValidationError,
    pattern,
)

is_int = InstanceOf(int)


def test_argument_repr():
    argument = Argument(is_int, typehint=int, default=None)
    assert repr(argument) == (
        "Argument(validator=InstanceOf(type=<class 'int'>), default=None, "
        "typehint=<class 'int'>)"
    )


def test_default_argument():
    annotation = Argument.default(validator=lambda x, context: int(x), default=3)
    assert annotation.validate(1) == 1
    with pytest.raises(TypeError):
        annotation.validate(None)


@pytest.mark.parametrize(
    ('default', 'expected'),
    [(None, None), (0, 0), ('default', 'default')],
)
def test_optional_argument(default, expected):
    annotation = Argument.optional(default=default)
    assert annotation.validate(None) == expected


@pytest.mark.parametrize(
    ('argument', 'value', 'expected'),
    [
        (Argument.optional(Any(), default=None), None, None),
        (Argument.optional(Any(), default=None), 'three', 'three'),
        (Argument.optional(Any(), default=1), None, 1),
        (Argument.optional(CoercedTo(int), default=11), None, 11),
        (Argument.optional(CoercedTo(int), default=None), None, None),
        (Argument.optional(CoercedTo(int), default=None), 18, 18),
        (Argument.optional(CoercedTo(str), default=None), 'caracal', 'caracal'),
    ],
)
def test_valid_optional(argument, value, expected):
    assert argument.validate(value) == expected


@pytest.mark.parametrize(
    ('arg', 'value', 'expected'),
    [
        (Argument.optional(is_int, default=''), None, TypeError),
        (Argument.optional(is_int), 'lynx', TypeError),
    ],
)
def test_invalid_optional_argument(arg, value, expected):
    with pytest.raises(expected):
        arg(value)


def test_initialized():
    class Foo:
        a = 10

    field = Attribute.default(lambda self: self.a + 10)
    assert field == field

    assert field.initialize(Foo) == 20

    field2 = Attribute(validator=lambda x, this: str(x), default=lambda self: self.a)
    assert field != field2
    assert field2.initialize(Foo) == '10'


def test_parameter():
    def fn(x, this):
        return int(x) + this['other']

    annot = Argument.required(fn)
    p = Parameter('test', annotation=annot)

    assert p.annotation is annot
    assert p.default is inspect.Parameter.empty
    assert p.annotation.validate('2', {'other': 1}) == 3

    with pytest.raises(TypeError):
        p.annotation.validate({}, valid=inspect.Parameter.empty)

    ofn = Argument.optional(fn)
    op = Parameter('test', annotation=ofn)
    assert op.annotation._validator == Option(fn, default=None)
    assert op.default is None
    assert op.annotation.validate(None, {'other': 1}) is None

    with pytest.raises(TypeError, match="annotation must be an instance of Argument"):
        Parameter("wrong", annotation=Attribute(lambda x, context: x))


def test_signature():
    def to_int(x, this):
        return int(x)

    def add_other(x, this):
        return int(x) + this['other']

    other = Parameter('other', annotation=Argument.required(to_int))
    this = Parameter('this', annotation=Argument.required(add_other))

    sig = Signature(parameters=[other, this])
    assert sig.validate(1, 2) == {'other': 1, 'this': 3}
    assert sig.validate(other=1, this=2) == {'other': 1, 'this': 3}
    assert sig.validate(this=2, other=1) == {'other': 1, 'this': 3}


def test_signature_from_callable():
    def test(a: int, b: int, c: int = 1):
        ...

    sig = Signature.from_callable(test)
    assert sig.validate(2, 3) == {'a': 2, 'b': 3, 'c': 1}

    with pytest.raises(ValidationError):
        sig.validate(2, 3, "4")

    args, kwargs = sig.unbind(sig.validate(2, 3))
    assert args == (2, 3, 1)
    assert kwargs == {}


def test_signature_from_callable_with_varargs():
    def test(a: int, b: int, *args: int):
        ...

    sig = Signature.from_callable(test)
    assert sig.validate(2, 3) == {'a': 2, 'b': 3, 'args': ()}
    assert sig.validate(2, 3, 4) == {'a': 2, 'b': 3, 'args': (4,)}
    assert sig.validate(2, 3, 4, 5) == {'a': 2, 'b': 3, 'args': (4, 5)}
    assert sig.parameters['a'].annotation._typehint is int
    assert sig.parameters['b'].annotation._typehint is int
    assert sig.parameters['args'].annotation._typehint is int

    with pytest.raises(ValidationError):
        sig.validate(2, 3, 4, "5")

    args, kwargs = sig.unbind(sig.validate(2, 3, 4, 5))
    assert args == (2, 3, 4, 5)
    assert kwargs == {}


def test_signature_from_callable_with_positional_only_arguments():
    def test(a: int, b: int, /, c: int = 1):
        ...

    sig = Signature.from_callable(test)
    assert sig.validate(2, 3) == {'a': 2, 'b': 3, 'c': 1}
    assert sig.validate(2, 3, 4) == {'a': 2, 'b': 3, 'c': 4}
    assert sig.validate(2, 3, c=4) == {'a': 2, 'b': 3, 'c': 4}

    msg = "'b' parameter is positional only, but was passed as a keyword"
    with pytest.raises(TypeError, match=msg):
        sig.validate(1, b=2)

    args, kwargs = sig.unbind(sig.validate(2, 3))
    assert args == (2, 3, 1)
    assert kwargs == {}


def test_signature_from_callable_with_keyword_only_arguments():
    def test(a: int, b: int, *, c: float, d: float = 0.0):
        ...

    sig = Signature.from_callable(test)
    assert sig.validate(2, 3, c=4.0) == {'a': 2, 'b': 3, 'c': 4.0, 'd': 0.0}
    assert sig.validate(2, 3, c=4.0, d=5.0) == {'a': 2, 'b': 3, 'c': 4.0, 'd': 5.0}

    with pytest.raises(TypeError, match="missing a required argument: 'c'"):
        sig.validate(2, 3)
    with pytest.raises(TypeError, match="too many positional arguments"):
        sig.validate(2, 3, 4)

    args, kwargs = sig.unbind(sig.validate(2, 3, c=4.0))
    assert args == (2, 3)
    assert kwargs == {'c': 4.0, 'd': 0.0}


def test_signature_unbind():
    def to_int(x, this):
        return int(x)

    def add_other(x, this):
        return int(x) + this['other']

    other = Parameter('other', annotation=Argument.required(to_int))
    this = Parameter('this', annotation=Argument.required(add_other))

    sig = Signature(parameters=[other, this])
    params = sig.validate(1, this=2)

    args, kwargs = sig.unbind(params)
    assert args == (1, 3)
    assert kwargs == {}


a = Parameter('a', annotation=Argument.required(CoercedTo(float)))
b = Parameter('b', annotation=Argument.required(CoercedTo(float)))
c = Parameter('c', annotation=Argument.default(default=0, validator=CoercedTo(float)))
d = Parameter(
    'd',
    annotation=Argument.default(default=tuple(), validator=TupleOf(CoercedTo(float))),
)
e = Parameter('e', annotation=Argument.optional(validator=CoercedTo(float)))
sig = Signature(parameters=[a, b, c, d, e])


@pytest.mark.parametrize('d', [(), (5, 6, 7)])
def test_signature_unbind_with_empty_variadic(d):
    params = sig.validate(1, 2, 3, d, e=4)
    assert params == {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': d, 'e': 4.0}

    args, kwargs = sig.unbind(params)
    assert args == (1.0, 2.0, 3.0, tuple(map(float, d)), 4.0)
    assert kwargs == {}

    params_again = sig.validate(*args, **kwargs)
    assert params_again == params


def test_annotated_function():
    @annotated(a=InstanceOf(int), b=InstanceOf(int), c=InstanceOf(int))
    def test(a, b, c=1):
        return a + b + c

    assert test(2, 3) == 6
    assert test(2, 3, 4) == 9
    assert test(2, 3, c=4) == 9
    assert test(a=2, b=3, c=4) == 9

    with pytest.raises(ValidationError):
        test(2, 3, c='4')

    @annotated(a=InstanceOf(int))
    def test(a, b, c=1):
        return (a, b, c)

    assert test(2, "3") == (2, "3", 1)


def test_annotated_function_with_type_annotations():
    @annotated()
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    assert test(2, 3) == 6

    @annotated
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    assert test(2, 3) == 6

    @annotated
    def test(a: int, b, c=1):
        return (a, b, c)

    assert test(2, 3, "4") == (2, 3, "4")


def test_annotated_function_with_return_type_annotation():
    @annotated
    def test_ok(a: int, b: int, c: int = 1) -> int:
        return a + b + c

    @annotated
    def test_wrong(a: int, b: int, c: int = 1) -> int:
        return "invalid result"

    assert test_ok(2, 3) == 6
    with pytest.raises(ValidationError):
        test_wrong(2, 3)


def test_annotated_function_with_keyword_overrides():
    @annotated(b=InstanceOf(float))
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    with pytest.raises(ValidationError):
        test(2, 3)

    assert test(2, 3.0) == 6.0


def test_annotated_function_with_list_overrides():
    @annotated([InstanceOf(int), InstanceOf(int), InstanceOf(float)])
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    with pytest.raises(ValidationError):
        test(2, 3, 4)


def test_annotated_function_with_list_overrides_and_return_override():
    @annotated([InstanceOf(int), InstanceOf(int), InstanceOf(float)], InstanceOf(float))
    def test(a: int, b: int, c: int = 1):
        return a + b + c

    with pytest.raises(ValidationError):
        test(2, 3, 4)

    assert test(2, 3, 4.0) == 9.0


@pattern
def short_str(x, this):
    if len(x) > 3:
        return x
    else:
        return NoMatch


@pattern
def endswith_d(x, this):
    if x.endswith('d'):
        return x
    else:
        return NoMatch


def test_annotated_function_with_complex_type_annotations():
    @annotated
    def test(a: Annotated[str, short_str, endswith_d], b: Union[int, float]):
        return a, b

    assert test("abcd", 1) == ("abcd", 1)
    assert test("---d", 1.0) == ("---d", 1.0)

    with pytest.raises(ValidationError, match="doesn't match"):
        test("---c", 1)
    with pytest.raises(ValidationError, match="doesn't match"):
        test("123", 1)
    with pytest.raises(ValidationError, match="'qweqwe' doesn't match"):
        test("abcd", "qweqwe")


def test_annotated_function_without_annotations():
    @annotated
    def test(a, b, c):
        return a, b, c

    assert test(1, 2, 3) == (1, 2, 3)
    assert test.__signature__.parameters.keys() == {'a', 'b', 'c'}


def test_annotated_function_without_decoration():
    def test(a, b, c):
        return a + b + c

    func = annotated(test)
    with pytest.raises(TypeError):
        func(1, 2)

    assert func(1, 2, c=3) == 6


def test_annotated_function_with_varargs():
    @annotated
    def test(a: float, b: float, *args: int):
        return sum((a, b) + args)

    assert test(1.0, 2.0, 3, 4) == 10.0
    assert test(1.0, 2.0, 3, 4, 5) == 15.0

    with pytest.raises(ValidationError):
        test(1.0, 2.0, 3, 4, 5, 6.0)


def test_annotated_function_with_varkwargs():
    @annotated
    def test(a: float, b: float, **kwargs: int):
        return sum((a, b) + tuple(kwargs.values()))

    assert test(1.0, 2.0, c=3, d=4) == 10.0
    assert test(1.0, 2.0, c=3, d=4, e=5) == 15.0

    with pytest.raises(ValidationError):
        test(1.0, 2.0, c=3, d=4, e=5, f=6.0)
