from __future__ import annotations

import ibis
from ibis.expr.types.relations import DerefMap

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


def test_dereference_project():
    p = t.select([t.int_col, t.double_col])

    mapping = DerefMap.from_targets([p.op()])
    expected = dereference_expect(
        {
            p.int_col: p.int_col,
            p.double_col: p.double_col,
            t.int_col: p.int_col,
            t.double_col: p.double_col,
        }
    )
    assert mapping.subs == expected


def test_dereference_mapping_self_reference():
    v = t.view()

    mapping = DerefMap.from_targets([v.op()])
    expected = dereference_expect(
        {
            v.int_col: v.int_col,
            v.double_col: v.double_col,
            v.string_col: v.string_col,
        }
    )
    assert mapping.subs == expected
