from matchpy import (
    Arity,
    Atom,
    Operation,
    Pattern,
    Symbol,
    Wildcard,
    match,
    substitute,
)

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.common.validators import noop, validator

# literal_types = (
#     BaseGeometry,
#     bytes,
#     datetime.date,
#     datetime.datetime,
#     datetime.time,
#     datetime.timedelta,
#     enum.Enum,
#     float,
#     frozenset,
#     int,
#     frozendict,
#     np.generic,
#     np.ndarray,
#     pd.Timedelta,
#     pd.Timestamp,
#     str,
#     tuple,
#     type(None),
#     uuid.UUID,
#     decimal.Decimal,
# )

# 1. rules should allow operations and eventually convert expressions to operations
# 2. add Node.register(Wildcard) so isinstance(wildcard, Node) should work and bypass validation


@validator
def noop(obj, **kwargs):
    return obj


class Node(ops.Node):
    def __init_subclass__(
        cls,
        /,
        name=None,
        arity=False,
        associative=False,
        commutative=False,
        one_identity=False,
        infix=False,
        **kwargs,
    ):
        # TODO(kszucs): raise if class already has these attributes
        cls.name = name or cls.__name__
        cls.arity = arity or Arity(len(cls.argnames), True)
        cls.associative = associative
        cls.commutative = commutative
        cls.one_identity = one_identity
        cls.infix = infix

    # TODO(kszucs): may need to port more dunder methods from matchpy.Operation
    # like __getitem__ or __contains__

    # def __init__(self, **kwargs):
    #     for value in kwargs.values():
    #         assert isinstance(
    #             value, (Node, Atom, tuple) + literal_types
    #         )
    #     super().__init__(**kwargs)

    def __iter__(self):
        return iter(self.args)

    def __len__(self):
        return len(self.args)

    def __getitem__(self, key):
        print("===================")
        return key

    def __contains__(self, key):
        print("================")
        return True

    @property
    def operands(self):
        print("#############################")
        return self.args

    def collect_variables(self):
        print("$")

    def collect_symbols(self, symbols):
        print("6666")


class Literal(Node):
    # TODO(kszucs): will need to add some more tooling to allow symbols as inputs
    value = noop
    dtype = noop


class Binary(Node):
    right = noop
    left = noop


class Add(Binary, commutative=True, associative=True):
    pass


Operation.register(Node)

# TODO(kszucs): make literal and datatype symbols?


def test_valami():
    a = Literal(value=1, dtype=int)
    b = Literal(value=2, dtype=int)
    c = Literal(value=0, dtype=int)

    x = Wildcard.dot('x')
    y = Wildcard.dot('y')
    z = Wildcard.dot('z')

    subject = Add(a, Add(b, c))
    pattern = Pattern(Add(Add(x, z), y))

    substitution = next(match(subject, pattern))
    from pprint import pprint

    pprint(substitution)


test_valami()
