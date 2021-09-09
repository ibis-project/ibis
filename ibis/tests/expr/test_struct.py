from collections import OrderedDict

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.tests.util import assert_pickle_roundtrip


def test_struct_operations():
    value = OrderedDict(
        [
            ('a', 1),
            ('b', list('abc')),
            ('c', OrderedDict([('foo', [1.0, 2.0])])),
        ]
    )
    expr = ibis.literal(value)
    assert isinstance(expr, ir.StructValue)
    assert isinstance(expr['b'], ir.ArrayValue)
    assert isinstance(expr['a'].op(), ops.StructField)


def test_struct_field_dir():
    t = ibis.table([('struct_col', 'struct<my_field: string>')])
    assert 'struct_col' in dir(t)
    assert 'my_field' in dir(t.struct_col)


def test_struct_pickle():
    struct_scalar_expr = ibis.literal(
        OrderedDict([("fruit", "pear"), ("weight", 0)])
    )

    assert_pickle_roundtrip(struct_scalar_expr)
