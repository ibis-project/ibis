from __future__ import annotations

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.base.sqlglot.rewrites import Select, sqlize


def test_sqlize():
    t = ibis.table(
        name="t",
        schema={
            "a": dt.int64,
            "b": dt.string,
            "c": dt.double,
            "d": dt.boolean,
        },
    )

    t = t.mutate(e=t.a.fillna(0))
    t = t.filter(t.a > 0)
    t = t.order_by(t.b)
    t = t.mutate(f=t.a + 1)
    s = sqlize(t.op())

    assert isinstance(s, Select)
