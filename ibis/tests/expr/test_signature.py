import pickle
from inspect import Signature

import pytest
from toolz import identity

import ibis.expr.rules as rlz
from ibis.common.exceptions import IbisTypeError
from ibis.expr.signature import (
    Annotable,
    Argument,
    Optional,
    Parameter,
    Validator,
)


class IsInt(Validator):
    def __call__(self, arg, **kwargs):
        if not isinstance(arg, int):
            raise TypeError(int)
        return arg


class IsFloat(Validator):
    def __call__(self, arg, **kwargs):
        if not isinstance(arg, float):
            raise TypeError(float)
        return arg


class Op(Annotable):
    __slots__ = ('_cache', '_hash')


class ValueOp(Op):
    arg = Argument(object)


class StringOp(ValueOp):
    arg = Argument(str)


class MagicString(StringOp):
    foo = Argument(str)
    bar = Argument(bool)
    baz = Argument(int)


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
    assert validator(None) == expected


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


def test_signature_inheritance():
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

    assert IntBinop.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt()),
            Parameter('right', validator=IsInt()),
        ]
    )

    assert FloatAddRhs.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt()),
            Parameter('right', validator=IsFloat()),
        ]
    )

    assert FloatAddClip.__signature__ == Signature(
        [
            Parameter('left', validator=IsFloat()),
            Parameter('right', validator=IsFloat()),
            Parameter('clip_lower', validator=Optional(IsInt(), default=0)),
            Parameter('clip_upper', validator=Optional(IsInt(), default=10)),
        ]
    )

    assert IntAddClip.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt()),
            Parameter('right', validator=IsInt()),
            Parameter('clip_lower', validator=Optional(IsInt(), default=0)),
            Parameter('clip_upper', validator=Optional(IsInt(), default=10)),
        ]
    )


def test_positional_argument_reordering():
    class Farm(Annotable):
        ducks = IsInt()
        donkeys = IsInt()
        horses = IsInt()
        goats = IsInt()
        chickens = IsInt()

    class NoHooves(Farm):
        horses = Optional(IsInt(), default=0)
        goats = Optional(IsInt(), default=0)
        donkeys = Optional(IsInt(), default=0)

    f1 = Farm(1, 2, 3, 4, 5)
    f2 = Farm(1, 2, goats=4, chickens=5, horses=3)
    f3 = Farm(1, 0, 0, 0, 100)
    assert f1 == f2
    assert f1 != f3

    g1 = NoHooves(1, 2, donkeys=-1)
    assert g1.ducks == 1
    assert g1.chickens == 2
    assert g1.donkeys == -1
    assert g1.horses == 0
    assert g1.goats == 0


def test_copy_default():
    default = []

    class Op(Annotable):
        arg = rlz.optional(rlz.instance_of(list), default=default)

    op = Op()
    assert op.arg is not default


def test_slots_are_inherited_and_overridable():
    class Op(Annotable):
        __slots__ = ('_cache',)  # first definition
        arg = Argument(lambda x: x)

    class StringOp(Op):
        arg = Argument(str)  # new overridden slot

    class StringSplit(StringOp):
        sep = Argument(str)  # new slot

    class StringJoin(StringOp):
        __slots__ = ('_memoize',)  # new slot
        sep = Argument(str)  # new overridden slot

    assert Op.__slots__ == ('_cache', 'arg')
    assert StringOp.__slots__ == ('arg',)
    assert StringSplit.__slots__ == ('sep',)
    assert StringJoin.__slots__ == ('_memoize', 'sep')


def test_multiple_inheritance():
    # multiple inheritance is allowed only if one of the parents has non-empty
    # __slots__ definition, otherwise python will raise lay-out conflict

    class Op(Annotable):
        __slots__ = ('_hash',)

    class ValueOp(Annotable):
        arg = Argument(object)

    class Reduction(ValueOp):
        _reduction = True

    class UDF(ValueOp):
        func = Argument(lambda fn, this: fn)

    class UDAF(UDF, Reduction):
        arity = Argument(int)

    class A(Annotable):
        a = Argument(int)

    class B(Annotable):
        b = Argument(int)

    msg = "multiple bases have instance lay-out conflict"
    with pytest.raises(TypeError, match=msg):

        class AB(A, B):
            ab = Argument(int)

    assert UDAF.__slots__ == ('arity',)
    strlen = UDAF(arg=2, func=lambda value: len(str(value)), arity=1)
    assert strlen.arg == 2
    assert strlen.arity == 1
    assert strlen._reduction is True


def test_pickling_support():
    op = MagicString(arg="something", foo="magic", bar=True, baz=8)
    raw = pickle.dumps(op)
    loaded = pickle.loads(raw)
    assert op == loaded
