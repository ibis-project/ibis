from __future__ import annotations

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from ibis.expr.newrels import Field, UnboundTable

t = UnboundTable(
    name="t",
    schema={
        "bool_col": "boolean",
        "int_col": "int64",
        "float_col": "float64",
        "string_col": "string",
    },
).to_expr()


def test_field():
    f = Field(t, "bool_col")
    assert f.rel == t.op()
    assert f.name == "bool_col"
    assert f.shape == ds.columnar
    assert f.dtype == dt.boolean
    assert f == t.bool_col
