"""
Constants for the arrow backend.
"""
import operator
import ibis.expr.operations as ops
import ibis

LEFT_JOIN_SUFFIX = '_ibis_left_{}'.format(ibis.util.guid())
RIGHT_JOIN_SUFFIX = '_ibis_right_{}'.format(ibis.util.guid())
JOIN_SUFFIXES = LEFT_JOIN_SUFFIX, RIGHT_JOIN_SUFFIX
ALTERNATE_SUFFIXES = {
    LEFT_JOIN_SUFFIX: RIGHT_JOIN_SUFFIX,
    RIGHT_JOIN_SUFFIX: LEFT_JOIN_SUFFIX,
}

BINARY_OPERATIONS = {
    ops.Greater: operator.gt,
    ops.Less: operator.lt,
    ops.LessEqual: operator.le,
    ops.GreaterEqual: operator.ge,
    ops.Equals: operator.eq,
    ops.NotEquals: operator.ne,

    ops.And: operator.and_,
    ops.Or: operator.or_,
    ops.Xor: operator.xor,

    ops.Add: operator.add,
    ops.Subtract: operator.sub,
    ops.Multiply: operator.mul,
    ops.Divide: operator.truediv,
    ops.FloorDivide: operator.floordiv,
    ops.Modulus: operator.mod,
    ops.Power: operator.pow,
    # TODO: implement identical operator for arrow
    # ops.IdenticalTo: lambda x, y: (x == y) | (pd.isnull(x) & pd.isnull(y))
}
