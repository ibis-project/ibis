from functools import partial
from inspect import Signature
from collections import OrderedDict

import pytest
from toolz import identity

from ibis.common.exceptions import IbisTypeError
from ibis.expr.signature import Annotable, Argument, Parameter, Optional, Validator, EMPTY


# @pytest.mark.parametrize(
#     ('validator', 'expected'),
#     [(lambda x: x, 3), (lambda x: x ** 2, 9), (lambda x: x + 1, 4)],
#     ids=['identity', 'square', 'inc'],
# )
# def test_argument(validator, expected):
#     arg = Argument(validator)

#     # test coercion
#     assert arg.validate(3) == expected

#     # syntactic sugar
#     assert arg(3) == expected


@pytest.mark.parametrize('validator', [3, 'coerce'])
def test_invalid_validator(validator):
    with pytest.raises(TypeError):
        Argument(validator)


def test_invalid_arity_validator():
    arg = Argument(lambda x, y: x + y)
    with pytest.raises(TypeError):
        arg('value')


def test_argument_raise_on_missing_value():
    validator = Argument(lambda x: x)

    expected_msg = "missing 1 required positional argument"
    with pytest.raises(TypeError, match=expected_msg):
        validator()

    expected_msg = "got an unexpected keyword argument 'name'"
    with pytest.raises(TypeError, match=expected_msg):
        validator(name='mandatory')


@pytest.mark.parametrize(
    ('default', 'expected'),
    [(None, None), (0, 0), ('default', 'default'), (lambda: 3, 3)],
)
def test_optional_argument(default, expected):
    validator = Optional(lambda x: x, default=default)
    assert validator(EMPTY) == expected


@pytest.mark.parametrize(
    ('validator', 'value', 'expected'),
    [
        (Argument(identity, default=None), None, None),
        (Argument(identity, default=None), 'three', 'three'),
        (Argument(identity, default=1), None, 1),
        (Argument(identity, default=lambda: 8), 'cat', 'cat'),
        (Argument(identity, default=lambda: 8), None, 8),
        (Argument(int, default=11), None, 11),
        (Argument(int, default=None), None, None),
        (Argument(int, default=None), 18, 18),
        (Argument(str, default=None), 'caracal', 'caracal'),
    ],
)
def test_valid_optional(validator, value, expected):
    assert validator(value) == expected


@pytest.mark.parametrize(
    ('arg', 'value', 'expected'),
    [
        (Argument(int, default=''), None, IbisTypeError),
        (Argument(int), 'lynx', IbisTypeError),
    ],
)
def test_invalid_optional(arg, value, expected):
    with pytest.raises(expected):
        arg(value)


# between = TypeSignature(
#     [
#         ('value', Argument(int)),
#         ('lower', Argument(int, default=0)),
#         ('upper', Argument(int, default=None)),
#     ]
# )


# @pytest.mark.parametrize(
#     ('call', 'expected'),
#     [
#         (partial(between, 3), (3, 0, None)),
#         (partial(between, 3), (3, 0, None)),
#         (partial(between, 3), (3, 0, None)),
#         (partial(between, 3, 1), (3, 1, None)),
#         (partial(between, 4, 2, 5), (4, 2, 5)),
#         (partial(between, 3, lower=1), (3, 1, None)),
#         (partial(between, 4, lower=2, upper=5), (4, 2, 5)),
#         (partial(between, 4, upper=5), (4, 0, 5)),
#         (partial(between, value=4, upper=5), (4, 0, 5)),
#     ],
# )
# def test_input_signature(call, expected):
#     assert call() == list(zip(['value', 'lower', 'upper'], expected))


def test_annotable():
    class Between(Annotable):
        value = Argument(int)
        lower = Argument(int, default=0)
        upper = Argument(int, default=None)

    argnames = ('value', 'lower', 'upper')
    signature = Between.__signature__
    assert isinstance(signature, Signature)
    assert tuple(signature.parameters.keys()) == argnames
    assert Between.__slots__ == argnames

    obj = Between(10, lower=2)
    assert obj.value == 10
    assert obj.lower == 2
    assert obj.upper is None

    assert obj.args == (10, 2, None)
    assert obj.argnames == argnames


def test_maintain_definition_order():
    class Between(Annotable):
        value = Argument(int)
        lower = Argument(int, default=0)
        upper = Argument(int, default=None)

    param_names = list(Between.__signature__.parameters.keys())
    assert param_names == ['value', 'lower', 'upper']


# def test_signature_equals():
#     s1 = TypeSignature([('left', Argument(int)), ('right', Argument(int))])
#     s2 = TypeSignature([('left', Argument(int)), ('right', Argument(int))])
#     s3 = TypeSignature([('left', Argument(int)), ('right', Argument(float))])
#     s4 = TypeSignature([('left', Argument(int)), ('right', Argument(float))])
#     s5 = TypeSignature(
#         [('left_one', Argument(int)), ('right', Argument(float))]
#     )
#     s6 = TypeSignature([('left_one', Argument(int)), ('right', Argument(int))])
#     assert s1 == s2
#     assert s3 == s4
#     assert s1 != s3
#     assert s2 != s4
#     assert s1 != s5
#     assert s2 != s6
#     assert s5 != s6


def test_signature_inheritance():
    class IsInt(Validator):
        def __call__(self, arg, **kwargs):
            return isinstance(arg, int)

    class IsFloat(Validator):
        def __call__(self, arg, **kwargs):
            return isinstance(arg, float)

    class IntBinop(Annotable):
        left = IsInt()
        right = IsInt()

    class FloatAddRhs(IntBinop):
        right = IsFloat()

    class FloatAddClip(FloatAddRhs):
        left = IsFloat()
        clip_lower = Optional(IsInt(), default=0)
        clip_upper = Optional(IsInt(), default=10)

    class IntAddClip(FloatAddClip, IntBinop):
        pass

    assert IntBinop.__signature__ == Signature([
        Parameter('left', validator=IsInt()),
        Parameter('right', validator=IsInt())
    ])

    assert FloatAddRhs.__signature__ == Signature([
        Parameter('left', validator=IsInt()),
        Parameter('right', validator=IsFloat())
    ])

    assert FloatAddClip.__signature__ == Signature([
        Parameter('left', validator=IsFloat()),
        Parameter('right', validator=IsFloat()),
        Parameter('clip_lower', validator=Optional(IsInt(), default=0)),
        Parameter('clip_upper', validator=Optional(IsInt(), default=10)),
    ])

    assert IntAddClip.__signature__ == Signature([
        Parameter('left', validator=IsInt()),
        Parameter('right', validator=IsInt()),
        Parameter('clip_lower', validator=Optional(IsInt(), default=0)),
        Parameter('clip_upper', validator=Optional(IsInt(), default=10)),
    ])


def test_slots_are_inherited_and_overridable():
    class Op(Annotable):
        __slots__ = ('_cache',)  # first definition
        arg = Argument(lambda x: x)

    class StringOp(Op):
        arg = Argument(str)  # inherit

    class StringSplit(StringOp):
        sep = Argument(str)  # inherit

    class StringJoin(StringOp):
        __slots__ = ('_memoize',)  # override
        sep = Argument(str)

    assert Op.__slots__ == ('_cache', 'arg')
    assert StringOp.__slots__ == ('_cache', 'arg')
    assert StringSplit.__slots__ == ('_cache', 'arg', 'sep')
    assert StringJoin.__slots__ == ('_memoize', 'arg', 'sep')
