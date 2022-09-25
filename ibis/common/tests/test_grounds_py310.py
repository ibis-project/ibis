from ibis.common.grounds import Annotable
from ibis.common.validators import instance_of

IsAny = instance_of(object)
IsBool = instance_of(bool)
IsFloat = instance_of(float)
IsInt = instance_of(int)
IsStr = instance_of(str)


class Node(Annotable):
    pass


class Literal(Node):
    value = instance_of((int, float, bool, str))
    dtype = instance_of(type)

    def __add__(self, other):
        return Add(self, other)


class BinaryOperation(Annotable):
    left = instance_of(Node)
    right = instance_of(Node)


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
