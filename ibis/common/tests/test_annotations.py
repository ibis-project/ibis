from __future__ import annotations

import inspect
from typing import Annotated, Union

import pytest

from ibis.common.annotations import (
    Argument,
    Attribute,
    Parameter,
    Signature,
    ValidationError,
    annotated,
    argument,
    attribute,
    optional,
)
from ibis.common.patterns import (
    Any,
    CoercedTo,
    InstanceOf,
    NoMatch,
    Option,
    TupleOf,
    pattern,
)

is_int = InstanceOf(int)


def test_argument_factory():
    a = argument(is_int, default=1, typehint=int)
    assert a == Argument(is_int, default=1, typehint=int)

    a = argument(is_int, default=1)
    assert a == Argument(is_int, default=1)

    a = argument(is_int)
    assert a == Argument(is_int)


def test_attribute_factory():
    a = attribute(is_int, default=1)
    assert a == Attribute(is_int, default=1)

    a = attribute(is_int)
    assert a == Attribute(is_int)

    a = attribute(default=2)
    assert a == Attribute(default=2)

    a = attribute(int, default=2)
    assert a == Attribute(int, default=2)


def test_annotations_are_immutable():
    a = argument(is_int, default=1)
    with pytest.raises(AttributeError):
        a.pattern = Any()
    with pytest.raises(AttributeError):
        a.default = 2

    a = attribute(is_int, default=1)
    with pytest.raises(AttributeError):
        a.pattern = Any()
    with pytest.raises(AttributeError):
        a.default = 2


def test_annotations_are_not_hashable():
    # in order to use the with mutable defaults
    a = argument(is_int, default=1)
    with pytest.raises(TypeError, match="unhashable type: 'Argument'"):
        hash(a)

    a = attribute(is_int, default=1)
    with pytest.raises(TypeError, match="unhashable type: 'Attribute'"):
        hash(a)


def test_argument_repr():
    argument = Argument(is_int, typehint=int, default=None)
    assert repr(argument) == (
        "Argument(pattern=InstanceOf(type=<class 'int'>), default=None, "
        "typehint=<class 'int'>, kind=<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>)"
    )


def test_default_argument():
    annotation = Argument(pattern=lambda x, context: int(x), default=3)
    assert annotation.pattern.match(1, {}) == 1


@pytest.mark.parametrize(
    ("default", "expected"),
    [(None, None), (0, 0), ("default", "default")],
)
def test_optional_argument(default, expected):
    annotation = optional(default=default)
    assert annotation.pattern.match(None, {}) == expected


@pytest.mark.parametrize(
    ("argument", "value", "expected"),
    [
        (optional(Any(), default=None), None, None),
        (optional(Any(), default=None), "three", "three"),
        (optional(Any(), default=1), None, 1),
        (optional(CoercedTo(int), default=11), None, 11),
        (optional(CoercedTo(int), default=None), None, None),
        (optional(CoercedTo(int), default=None), 18, 18),
        (optional(CoercedTo(str), default=None), "caracal", "caracal"),
    ],
)
def test_valid_optional(argument, value, expected):
    assert argument.pattern.match(value, {}) == expected


def test_attribute_default_value():
    class Foo:
        a = 10

    assert not Attribute().has_default()

    field = Attribute(default=lambda self: self.a + 10)
    assert field.has_default()
    assert field == field

    assert field.get_default("b", Foo) == 20

    field2 = Attribute(pattern=lambda x, this: str(x), default=lambda self: self.a)
    assert field2.has_default()
    assert field != field2
    assert field2.get_default("b", Foo) == "10"


def test_parameter():
    def fn(x, this):
        return int(x) + this["other"]

    annot = argument(fn)
    p = Parameter("test", annotation=annot)

    assert p.annotation is annot
    assert p.default is inspect.Parameter.empty
    assert p.annotation.pattern.match("2", {"other": 1}) == 3

    ofn = optional(fn)
    op = Parameter("test", annotation=ofn)
    assert op.annotation.pattern == Option(fn, default=None)
    assert op.default is None
    assert op.annotation.pattern.match(None, {"other": 1}) is None

    with pytest.raises(TypeError, match="annotation must be an instance of Argument"):
        Parameter("wrong", annotation=Attribute(lambda x, context: x))


def test_signature():
    def to_int(x, this):
        return int(x)

    def add_other(x, this):
        return int(x) + this["other"]

    other = Parameter("other", annotation=Argument(to_int))
    this = Parameter("this", annotation=Argument(add_other))

    sig = Signature(parameters=[other, this])
    assert sig.validate(None, args=(1, 2), kwargs={}) == {"other": 1, "this": 3}
    assert sig.validate(None, args=(), kwargs=dict(other=1, this=2)) == {
        "other": 1,
        "this": 3,
    }
    assert sig.validate(None, args=(), kwargs=dict(this=2, other=1)) == {
        "other": 1,
        "this": 3,
    }


def test_signature_from_callable():
    def test(a: int, b: int, c: int = 1):
        ...

    sig = Signature.from_callable(test)
    assert sig.validate(test, args=(2, 3), kwargs={}) == {"a": 2, "b": 3, "c": 1}

    with pytest.raises(ValidationError):
        sig.validate(test, args=(2, 3, "4"), kwargs={})

    args, kwargs = sig.unbind(sig.validate(test, args=(2, 3), kwargs={}))
    assert args == (2, 3, 1)
    assert kwargs == {}


def test_signature_from_callable_with_varargs():
    def test(a: int, b: int, *args: int):
        ...

    sig = Signature.from_callable(test)
    assert sig.validate(test, args=(2, 3), kwargs={}) == {"a": 2, "b": 3, "args": ()}
    assert sig.validate(test, args=(2, 3, 4), kwargs={}) == {
        "a": 2,
        "b": 3,
        "args": (4,),
    }
    assert sig.validate(test, args=(2, 3, 4, 5), kwargs={}) == {
        "a": 2,
        "b": 3,
        "args": (4, 5),
    }
    assert sig.parameters["a"].annotation.typehint is int
    assert sig.parameters["b"].annotation.typehint is int
    assert sig.parameters["args"].annotation.typehint is int

    with pytest.raises(ValidationError):
        sig.validate(test, args=(2, 3, 4, "5"), kwargs={})

    args, kwargs = sig.unbind(sig.validate(test, args=(2, 3, 4, 5), kwargs={}))
    assert args == (2, 3, 4, 5)
    assert kwargs == {}


def test_signature_from_callable_with_positional_only_arguments(snapshot):
    def test(a: int, b: int, /, c: int = 1):
        ...

    sig = Signature.from_callable(test)
    assert sig.validate(test, args=(2, 3), kwargs={}) == {"a": 2, "b": 3, "c": 1}
    assert sig.validate(test, args=(2, 3, 4), kwargs={}) == {"a": 2, "b": 3, "c": 4}
    assert sig.validate(test, args=(2, 3), kwargs=dict(c=4)) == {"a": 2, "b": 3, "c": 4}

    with pytest.raises(ValidationError) as excinfo:
        sig.validate(test, args=(1,), kwargs=dict(b=2))
    snapshot.assert_match(str(excinfo.value), "parameter_is_positional_only.txt")

    args, kwargs = sig.unbind(sig.validate(test, args=(2, 3), kwargs={}))
    assert args == (2, 3, 1)
    assert kwargs == {}


def test_signature_from_callable_with_keyword_only_arguments(snapshot):
    def test(a: int, b: int, *, c: float, d: float = 0.0):
        ...

    sig = Signature.from_callable(test)
    assert sig.validate(test, args=(2, 3), kwargs=dict(c=4.0)) == {
        "a": 2,
        "b": 3,
        "c": 4.0,
        "d": 0.0,
    }
    assert sig.validate(test, args=(2, 3), kwargs=dict(c=4.0, d=5.0)) == {
        "a": 2,
        "b": 3,
        "c": 4.0,
        "d": 5.0,
    }

    with pytest.raises(ValidationError) as excinfo:
        sig.validate(test, args=(2, 3), kwargs={})
    snapshot.assert_match(str(excinfo.value), "missing_a_required_argument.txt")

    with pytest.raises(ValidationError) as excinfo:
        sig.validate(test, args=(2, 3, 4), kwargs={})
    snapshot.assert_match(str(excinfo.value), "too_many_positional_arguments.txt")

    args, kwargs = sig.unbind(sig.validate(test, args=(2, 3), kwargs=dict(c=4.0)))
    assert args == (2, 3)
    assert kwargs == {"c": 4.0, "d": 0.0}


def test_signature_unbind():
    def to_int(x, this):
        return int(x)

    def add_other(x, this):
        return int(x) + this["other"]

    other = Parameter("other", annotation=Argument(to_int))
    this = Parameter("this", annotation=Argument(add_other))

    sig = Signature(parameters=[other, this])
    params = sig.validate(None, args=(1,), kwargs=dict(this=2))

    args, kwargs = sig.unbind(params)
    assert args == (1, 3)
    assert kwargs == {}


a = Parameter("a", annotation=Argument(CoercedTo(float)))
b = Parameter("b", annotation=Argument(CoercedTo(float)))
c = Parameter("c", annotation=Argument(CoercedTo(float), default=0))
d = Parameter(
    "d",
    annotation=Argument(TupleOf(CoercedTo(float)), default=()),
)
e = Parameter("e", annotation=Argument(Option(CoercedTo(float)), default=None))
sig = Signature(parameters=[a, b, c, d, e])


@pytest.mark.parametrize("d", [(), (5, 6, 7)])
def test_signature_unbind_with_empty_variadic(d):
    params = sig.validate(None, args=(1, 2, 3, d), kwargs=dict(e=4))
    assert params == {"a": 1.0, "b": 2.0, "c": 3.0, "d": d, "e": 4.0}

    args, kwargs = sig.unbind(params)
    assert args == (1.0, 2.0, 3.0, tuple(map(float, d)), 4.0)
    assert kwargs == {}

    params_again = sig.validate(None, args=args, kwargs=kwargs)
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
        test(2, 3, c="4")

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
    if x.endswith("d"):
        return x
    else:
        return NoMatch


def test_annotated_function_with_complex_type_annotations():
    @annotated
    def test(a: Annotated[str, short_str, endswith_d], b: Union[int, float]):
        return a, b

    assert test("abcd", 1) == ("abcd", 1)
    assert test("---d", 1.0) == ("---d", 1.0)

    with pytest.raises(ValidationError):
        test("---c", 1)
    with pytest.raises(ValidationError):
        test("123", 1)
    with pytest.raises(ValidationError):
        test("abcd", "qweqwe")


def test_annotated_function_without_annotations():
    @annotated
    def test(a, b, c):
        return a, b, c

    assert test(1, 2, 3) == (1, 2, 3)
    assert test.__signature__.parameters.keys() == {"a", "b", "c"}


def test_annotated_function_without_decoration(snapshot):
    def test(a, b, c):
        return a + b + c

    func = annotated(test)
    with pytest.raises(ValidationError) as excinfo:
        func(1, 2)
    snapshot.assert_match(str(excinfo.value), "error.txt")

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


def test_multiple_validation_failures():
    @annotated
    def test(a: float, b: float, *args: int, **kwargs: int):
        ...

    with pytest.raises(ValidationError) as excinfo:
        test(1.0, 2.0, 3.0, 4, c=5.0, d=6)

    assert len(excinfo.value.errors) == 2
