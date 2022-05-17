from matchpy import (
    Arity,
    Operation,
    Pattern,
    Symbol,
    Wildcard,
    match,
    substitute,
)

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.common.validators import noop


class MyOp(ops.Node):
    a = noop
    b = noop


# def test_instance_of_operation():
#     MyOp(1, 2)


def test_pina():
    f = Operation.new('f', Arity.binary)
    a = Symbol('a')
    print(f(a, a))

    x = Wildcard.dot('x')
    print(Pattern(f(a, x)))

    y = Wildcard.dot('y')
    b = Symbol('b')
    subject = f(a, b)
    pattern = Pattern(f(x, y))

    substitution = next(match(subject, pattern))
    print(substitution)

    print(substitute(pattern, substitution))

    z = Wildcard.plus('z')
    pattern = Pattern(f(z))
    subject = f(a, b)
    substitution = next(match(subject, pattern))
    print(substitution)


# def test_valami():
#     a = Symbol('a')
#     b = Symbol('b')
#     x = Wildcard.dot('x')
#     y = Wildcard.dot('y')

#     subject = MyOp(a, b)
#     pattern = Pattern(MyOp(x, y))

#     next(match(subject, pattern))
#     return
#     substitution = next(match(subject, pattern))
#     print(substitution)
