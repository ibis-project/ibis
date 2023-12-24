from __future__ import annotations

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.rewrites import Select, Window, sqlize

t = ibis.table(
    name="t",
    schema={
        "a": dt.int64,
        "b": dt.string,
        "c": dt.double,
        "d": dt.boolean,
    },
)


def test_sqlize():
    expr = t.mutate(e=t.a.fillna(0)).filter(t.a > 0).order_by(t.b).mutate(f=t.a + 1)

    result = sqlize(expr.op())
    expected = Select(
        parent=t,
        selections={
            "a": t.a,
            "b": t.b,
            "c": t.c,
            "d": t.d,
            "e": ops.Coalesce([t.a, 0]),
            "f": t.a + 1,
        },
        predicates=(t.a > 0,),
        sort_keys=(t.b.asc(),),
    )
    assert result == expected


def test_sqlize_dont_merge_windows():
    g = t.a.sum().name("g")
    h = t.a.cumsum().name("h")
    expr = t.mutate(g, h).filter(t.a > 0).select("a", "g", "h")

    result = sqlize(expr.op())
    sel1 = Select(
        parent=t,
        selections={
            "a": t.a,
            "b": t.b,
            "c": t.c,
            "d": t.d,
            "g": Window(how="rows", func=t.a.sum()),
            "h": Window(
                how="rows", func=t.a.sum(), end=ops.WindowBoundary(0, preceding=False)
            ),
        },
    ).to_expr()

    sel2 = Select(
        parent=sel1,
        selections={
            "a": sel1.a,
            "g": sel1.g,
            "h": sel1.h,
        },
        predicates=(sel1.a > 0,),
    )

    assert result == sel2
