from __future__ import annotations

import ibis
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


def test_struct_column_shape():
    one = ops.Literal(1, dtype=dt.int64)
    op = ops.StructColumn(names=("a",), values=(one,))

    assert op.shape == ds.scalar

    col = ops.TableColumn(
        ops.UnboundTable(schema=ibis.schema(dict(a="int64")), name="t"), "a"
    )
    op = ops.StructColumn(names=("a",), values=(col,))
    assert op.shape == ds.columnar
