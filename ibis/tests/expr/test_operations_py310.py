import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

one = ibis.literal(1)
two = ibis.literal(2)
three = ibis.literal(3)


def test_pattern_matching():
    match one.op():
        case ops.Literal(value, dtype=dtype):
            assert value == 1
            assert dtype is dt.int8
        case _:
            raise ValueError("Unable to match")

    match (one + two).op():
        case ops.Add(left, right):
            assert left == one.op()
            assert right == two.op()
        case _:
            raise ValueError("Unable to match")

    match (one + two).op():
        case ops.Add(right=right, left=left):
            assert left == one.op()
            assert right == two.op()
        case _:
            raise ValueError("Unable to match")
