from __future__ import annotations

import ibis
from ibis.expr.types.relations import dereference_mapping

t = ibis.table(
    [
        ("int_col", "int32"),
        ("double_col", "double"),
        ("string_col", "string"),
    ],
    name="t",
)


def dereference_expect(expected):
    return {k.op(): v.op() for k, v in expected.items()}


def test_dereference_mapping_self_reference():
    v = t.view()

    mapping = dereference_mapping([v.op()])
    expected = dereference_expect({})
    assert mapping == expected
