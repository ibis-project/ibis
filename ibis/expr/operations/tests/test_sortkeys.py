from __future__ import annotations

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


def test_sortkey_propagates_dtype_and_shape():
    k = ops.SortKey(ibis.literal(1), ascending=True)
    assert k.dtype == dt.int8
    assert k.shape.is_scalar()

    t = ibis.table([("a", "int16")], name="t")
    k = ops.SortKey(t.a, ascending=True)
    assert k.dtype == dt.int16
    assert k.shape.is_columnar()
