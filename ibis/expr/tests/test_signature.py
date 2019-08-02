from functools import partial

import pytest
from toolz import identity

from ibis.common.exceptions import IbisTypeError
from ibis.expr.signature import Annotable, Argument, TypeSignature


@pytest.mark.parametrize(
    ('validator', 'expected'),
    [(lambda x: x, 3), (lambda x: x ** 2, 9), (lambda x: x + 1, 4)],
    ids=['identity', 'square', 'inc'],
)
def test_argument(validator, expected):
    arg = Argument(validator)

    # test coercion
    assert arg.validate(3) == expected

    # syntactic sugar
    assert arg(3) == expected


@pytest.mark.parametrize('validator', [3, 'coerce'])
def test_invalid_validator(validator):
    with pytest.raises(TypeError):
        Argument(validator)


def test_invalid_arity_validator():
    arg = Argument(lambda x, y: x + y)
    with pytest.raises(TypeError):
        arg('value')


def test_argument_raise_on_missing_value():
    arg = Argument(lambda x: x)

    expected_msg = 'Missing required value for unnamed argument'
    with pytest.raises(TypeError, match=expected_msg):
        arg.validate()

    expected_msg = 'Missing required value for argument `mandatory`'
    with pytest.raises(TypeError, match=expected_msg):
        arg.validate(name='mandatory')


@pytest.mark.parametrize(
    ('default', 'expected'),
    [(None, None), (0, 0), ('default', 'default'), (lambda: 3, 3)],
)
def test_optional_argument(default, expected):
    arg = Argument(lambda x: x, default=default)
    assert arg.validate() == expected
    assert arg() == expected


@pytest.mark.parametrize(
    ('arg', 'value', 'expected'),
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
def test_valid_optional(arg, value, expected):
    assert arg(value) == expected


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


between = TypeSignature(
    [
        ('value', Argument(int)),
        ('lower', Argument(int, default=0)),
        ('upper', Argument(int, default=None)),
    ]
)


@pytest.mark.parametrize(
    ('call', 'expected'),
    [
        (partial(between, 3), (3, 0, None)),
        (partial(between, 3), (3, 0, None)),
        (partial(between, 3), (3, 0, None)),
        (partial(between, 3, 1), (3, 1, None)),
        (partial(between, 4, 2, 5), (4, 2, 5)),
        (partial(between, 3, lower=1), (3, 1, None)),
        (partial(between, 4, lower=2, upper=5), (4, 2, 5)),
        (partial(between, 4, upper=5), (4, 0, 5)),
        (partial(between, value=4, upper=5), (4, 0, 5)),
    ],
)
def test_input_signature(call, expected):
    assert call() == list(zip(['value', 'lower', 'upper'], expected))


def test_annotable():
    class Between(Annotable):
        value = Argument(int)
        lower = Argument(int, default=0)
        upper = Argument(int, default=None)

    argnames = ('value', 'lower', 'upper')
    assert isinstance(Between.signature, TypeSignature)
    assert Between.signature.names() == argnames
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

    assert list(Between.signature.keys()) == ['value', 'lower', 'upper']


def test_signature_equals():
    s1 = TypeSignature([('left', Argument(int)), ('right', Argument(int))])
    s2 = TypeSignature([('left', Argument(int)), ('right', Argument(int))])
    s3 = TypeSignature([('left', Argument(int)), ('right', Argument(float))])
    s4 = TypeSignature([('left', Argument(int)), ('right', Argument(float))])
    s5 = TypeSignature(
        [('left_one', Argument(int)), ('right', Argument(float))]
    )
    s6 = TypeSignature([('left_one', Argument(int)), ('right', Argument(int))])
    assert s1 == s2
    assert s3 == s4
    assert s1 != s3
    assert s2 != s4
    assert s1 != s5
    assert s2 != s6
    assert s5 != s6


def test_signature_inheritance():
    class IntBinop(Annotable):
        left = Argument(int)
        right = Argument(int)

    class FloatAddRhs(IntBinop):
        right = Argument(float)

    class FloatAddClip(FloatAddRhs):
        left = Argument(float)
        clip_lower = Argument(int, default=0)
        clip_upper = Argument(int, default=10)

    class IntAddClip(FloatAddClip, IntBinop):
        pass

    assert IntBinop.signature == TypeSignature(
        [('left', Argument(int)), ('right', Argument(int))]
    )
    assert FloatAddRhs.signature == TypeSignature(
        [('left', Argument(int)), ('right', Argument(float))]
    )
    assert FloatAddClip.signature == TypeSignature(
        [
            ('left', Argument(float)),
            ('right', Argument(float)),
            ('clip_lower', Argument(int, default=0)),
            ('clip_upper', Argument(int, default=10)),
        ]
    )
    assert IntAddClip.signature == TypeSignature(
        [
            ('left', Argument(int)),
            ('right', Argument(int)),
            ('clip_lower', Argument(int, default=0)),
            ('clip_upper', Argument(int, default=10)),
        ]
    )


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
