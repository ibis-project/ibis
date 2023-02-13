import pytest

import ibis
import ibis.expr.operations as ops
from ibis.common.exceptions import RelationError


class A:
    a: int
    b: str
    c: float


# identical to A
class B:
    a: int
    b: str
    c: float


# same as A with different order
class C:
    c: float
    b: str
    a: int


class D:
    a: str
    b: str
    c: str


a = ibis.table(A)
b = ibis.table(B)
c = ibis.table(C)
d = ibis.table(D)


@pytest.mark.parametrize('method', ['union', 'intersect', 'difference'])
def test_operation_requires_equal_schemas(method):
    with pytest.raises(RelationError):
        getattr(a, method)(d)


@pytest.mark.parametrize('method', ['union', 'intersect', 'difference'])
def test_operation_supports_schemas_with_different_field_order(method):
    u1 = getattr(a, method)(b)
    u2 = getattr(a, method)(c)

    assert u1.schema() == a.schema()
    assert u1.op().left == a.op()
    assert u1.op().right == b.op()

    # a selection is added to ensure that the field order of the right table
    # matches the field order of the left table
    assert u2.schema() == a.schema()
    assert u2.op().left == a.op()
    assert u2.op().right == ops.Selection(c.op(), ['a', 'b', 'c'])
