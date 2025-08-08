from __future__ import annotations

import pytest

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

    assert tuple(
        mapping.dereference(
            p.int_col.op(), p.double_col.op(), t.int_col.op(), t.double_col.op()
        )
    ) == (
        p.int_col.op(),
        p.double_col.op(),
        p.int_col.op(),
        p.double_col.op(),
    )


@pytest.mark.parametrize("column", ["int_col", "double_col", "string_col"])
def test_dereference_mapping_self_reference(column):
    v = t.view()

    mapping = DerefMap.from_targets([v.op()])
    assert tuple(mapping.dereference(v[column].op())) == (v[column].op(),)
