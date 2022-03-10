from inspect import Signature

import pytest
from toolz import identity

from ibis.expr.signature import (
    Annotable,
    Argument,
    Optional,
    Parameter,
    Validator,
)
from ibis.tests.util import assert_pickle_roundtrip


class ValidatorFunction(Validator):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class InstanceOf(Validator):
    def __init__(self, typ):
        self.typ = typ

    def __call__(self, arg, **kwargs):
        if not isinstance(arg, self.typ):
            raise TypeError(self.typ)
        return arg


IsAny = InstanceOf(object)
IsBool = InstanceOf(bool)
IsFloat = InstanceOf(float)
IsInt = InstanceOf(int)
IsStr = InstanceOf(str)


class Op(Annotable):
    __slots__ = ('_cache', '_hash')


class ValueOp(Op):
    arg = InstanceOf(object)


class StringOp(ValueOp):
    arg = InstanceOf(str)


class MagicString(StringOp):
    foo = Argument(str)
    bar = Argument(bool)
    baz = Argument(int)


def test_argument_is_deprecated():
    msg = r".*Argument.* is deprecated .* v3\.0; use Validator\."
    with pytest.warns(FutureWarning, match=msg):
        Argument(str)


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


def test_annotable():
    class Between(Annotable):
        value = IsInt
        lower = Optional(IsInt, default=0)
        upper = Optional(IsInt, default=None)

    class InBetween(Between):
        pass

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
    assert obj.__slots__ == ("value", "lower", "upper")
    assert not hasattr(obj, "__dict__")

    # test that a child without additional arguments doesn't have __dict__
    obj = InBetween(10, lower=2)
    assert obj.__slots__ == tuple()
    assert not hasattr(obj, "__dict__")


def test_maintain_definition_order():
    class Between(Annotable):
        value = IsInt
        lower = Optional(IsInt, default=0)
        upper = Optional(IsInt, default=None)

    param_names = list(Between.__signature__.parameters.keys())
    assert param_names == ['value', 'lower', 'upper']


def test_signature_inheritance():
    class IntBinop(Annotable):
        left = IsInt
        right = IsInt

    class FloatAddRhs(IntBinop):
        right = IsFloat

    class FloatAddClip(FloatAddRhs):
        left = IsFloat
        clip_lower = Optional(IsInt, default=0)
        clip_upper = Optional(IsInt, default=10)

    class IntAddClip(FloatAddClip, IntBinop):
        pass

    assert IntBinop.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt),
            Parameter('right', validator=IsInt),
        ]
    )

    assert FloatAddRhs.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt),
            Parameter('right', validator=IsFloat),
        ]
    )

    assert FloatAddClip.__signature__ == Signature(
        [
            Parameter('left', validator=IsFloat),
            Parameter('right', validator=IsFloat),
            Parameter('clip_lower', validator=Optional(IsInt, default=0)),
            Parameter('clip_upper', validator=Optional(IsInt, default=10)),
        ]
    )

    assert IntAddClip.__signature__ == Signature(
        [
            Parameter('left', validator=IsInt),
            Parameter('right', validator=IsInt),
            Parameter('clip_lower', validator=Optional(IsInt, default=0)),
            Parameter('clip_upper', validator=Optional(IsInt, default=10)),
        ]
    )


def test_positional_argument_reordering():
    class Farm(Annotable):
        ducks = IsInt
        donkeys = IsInt
        horses = IsInt
        goats = IsInt
        chickens = IsInt

    class NoHooves(Farm):
        horses = Optional(IsInt, default=0)
        goats = Optional(IsInt, default=0)
        donkeys = Optional(IsInt, default=0)

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
        arg = Optional(InstanceOf(list), default=default)

    op = Op()
    assert op.arg is not default


def test_slots_are_inherited_and_overridable():
    class Op(Annotable):
        __slots__ = ('_cache',)  # first definition
        arg = ValidatorFunction(lambda x: x)

    class StringOp(Op):
        arg = ValidatorFunction(str)  # new overridden slot

    class StringSplit(StringOp):
        sep = ValidatorFunction(str)  # new slot

    class StringJoin(StringOp):
        __slots__ = ('_memoize',)  # new slot
        sep = ValidatorFunction(str)  # new overridden slot

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
        arg = InstanceOf(object)

    class Reduction(ValueOp):
        _reduction = True

    class UDF(ValueOp):
        func = ValidatorFunction(lambda fn, this: fn)

    class UDAF(UDF, Reduction):
        arity = IsInt

    class A(Annotable):
        a = IsInt

    class B(Annotable):
        b = IsInt

    msg = "multiple bases have instance lay-out conflict"
    with pytest.raises(TypeError, match=msg):

        class AB(A, B):
            ab = IsInt

    assert UDAF.__slots__ == ('arity',)
    strlen = UDAF(arg=2, func=lambda value: len(str(value)), arity=1)
    assert strlen.arg == 2
    assert strlen.arity == 1
    assert strlen._reduction is True


@pytest.mark.parametrize(
    "obj",
    [
        MagicString(arg="something", foo="magic", bar=True, baz=8),
        Parameter("test"),
    ],
)
def test_pickling_support(obj):
    assert_pickle_roundtrip(obj)


def test_multiple_inheritance_argument_order():
    class ValueOp(Annotable):
        arg = IsAny

    class VersionedOp(ValueOp):
        version = IsInt

    class Reduction(Annotable):
        _reduction = True

    class Sum(VersionedOp, Reduction):
        where = Optional(IsBool, default=False)

    assert Sum._reduction is True
    assert str(Sum.__signature__) == "(arg, version, where=None)"


def test_multiple_inheritance_optional_argument_order():
    class ValueOp(Annotable):
        pass

    class ConditionalOp(Annotable):
        where = Optional(IsBool, default=False)

    class Between(ValueOp, ConditionalOp):
        min = IsInt
        max = IsInt
        how = Optional(IsStr, default="strict")

    assert str(Between.__signature__) == "(min, max, where=None, how=None)"
