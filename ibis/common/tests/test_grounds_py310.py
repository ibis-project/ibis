from sre_constants import AT_NON_BOUNDARY

from ibis.common.grounds import Annotable, Validator


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


class Node(Annotable):
    pass


class Literal(Node):
    value = InstanceOf((int, float, bool, str))
    dtype = InstanceOf(type)

    def __add__(self, other):
        return Add(self, other)


class BinaryOperation(Annotable):
    left = InstanceOf(Node)
    right = InstanceOf(Node)


class Add(BinaryOperation):
    pass


one = Literal(value=1, dtype=int)
two = Literal(value=2, dtype=int)


def test_pattern_matching():
    match one:
        case Literal(value, dtype=dtype):
            assert value == 1
            assert dtype is int
        case _:
            raise ValueError("Unable to match")

    match (one + two):
        case Add(left, right):
            assert left == one
            assert right == two
        case _:
            raise ValueError("Unable to match")
