from __future__ import annotations

import pytest

import ibis
import ibis.expr.operations as ops
from ibis.expr.rewrites import simplify

t = ibis.table(
    name="t",
    schema={
        "bool_col": "boolean",
        "int_col": "int64",
        "float_col": "float64",
        "string_col": "string",
    },
)
u = ibis.table({"int_col": "int64"}, name="u")


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(t.filter(t.int_col.isin(u.int_col)), id="in-table"),
        pytest.param(
            t.filter(t.int_col.isin(u.select(u.int_col).union(u).int_col)),
            id="in-union",
        ),
        pytest.param(
            t.filter(t.int_col.isin(u.select(u).view().int_col)), id="in-view"
        ),
        pytest.param(t.select(s=u.int_col.as_scalar()), id="scalar"),
        pytest.param(
            t.filter(ops.ExistsSubquery(u.select(u).op()).to_expr()), id="exists"
        ),
    ],
)
def test_simplify_keeps_subquery_relation(expr):
    # a subquery needs a relation with values, which a bare table, set
    # operation or view lacks
    ibis.to_sql(simplify(expr.op()).to_expr(), dialect="duckdb")


def test_simplify_collapses_subquery_relation_with_values():
    f = u.filter(u.int_col > 0)
    expr = t.filter(t.int_col.isin(f.int_col))
    (sub,) = simplify(expr.op()).find(ops.InSubquery)
    assert sub.rel == f.op()


v = ibis.table({"int_col": "int64", "arr": "array<int64>"}, name="v")

f = t.filter(t.int_col > 1)
fw = t.filter(t.int_col > t.int_col.mean().over())
p = t.select(t.int_col, t.string_col)
px = t.select(t.int_col, x=t.int_col + 1)
pw = t.select(t.int_col, avg=t.int_col.mean().over(group_by=t.string_col))
pc = t.select(t.int_col, cs=t.int_col.cumsum(order_by=t.int_col))
pe = t.select(t.int_col, flag=(u.int_col == t.int_col).any())
pr = t.select(t.int_col, r=ibis.random())
pu = v.select(v.int_col, x=v.arr.unnest())
pi = v.select(v.int_col, rid=v.rowid())


@pytest.mark.parametrize(
    "expr",
    [
        # a filter over a filter
        # safe to merge, since the plain predicate still runs first, but the
        # guard keeps every window apart
        pytest.param(
            f.filter(f.int_col > f.int_col.mean().over()), id="filters-outer-window"
        ),
        pytest.param(fw.filter(fw.int_col > 1), id="filters-inner-window"),
        pytest.param(f.filter((u.int_col == f.int_col).any()), id="filters-exists"),
        pytest.param(
            f.filter(f.int_col.isin(v.filter(v.int_col == f.int_col).int_col)),
            id="filters-isin",
        ),
        # a filter over a projection
        pytest.param(pc.filter(pc.cs > 5), id="reorder-window"),
        pytest.param(pw.filter(pw.int_col > 1), id="reorder-window-other-column"),
        pytest.param(pe.filter(pe.int_col > 1), id="reorder-inner-exists"),
        pytest.param(
            px.filter((u.int_col == px.int_col).any()), id="reorder-outer-exists"
        ),
        pytest.param(
            px.filter(px.int_col.isin(v.filter(v.int_col == px.int_col).int_col)),
            id="reorder-outer-isin",
        ),
        pytest.param(pr.filter(pr.r > 0.5), id="reorder-impure"),
        pytest.param(pu.filter(pu.x > 1), id="reorder-unnest"),
        pytest.param(pi.filter(pi.int_col > 1), id="reorder-rowid"),
        # a projection over a projection
        pytest.param(pw.select(pw.int_col, r=pw.avg.rank()), id="projects-window"),
        pytest.param(pr.select(a=pr.r + 1, b=pr.r + 2), id="projects-impure"),
        pytest.param(
            p.select(p.int_col, flag=(u.int_col == p.int_col).any()),
            id="projects-exists",
        ),
    ],
)
def test_simplify_keeps_blocked_relations_apart(expr):
    # merging would move a window, subquery, impure call, UNNEST or row id
    # across the relation it is tied to
    assert simplify(expr.op()) == expr.op()


def test_simplify_full_reprojection():
    t1 = t.select(t)
    t1_opt = simplify(t1.op())
    assert t1_opt == t.op()


def test_simplify_subsequent_field_selections():
    t1 = t.select(t.bool_col, t.int_col, t.float_col)
    assert t1.op() == ops.Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
            "float_col": t.float_col,
        },
    )

    t2 = t1.select(t1.bool_col, t1.int_col)
    t2_opt = simplify(t2.op())
    assert t2_opt == ops.Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
        },
    )

    t3 = t2.select(t2.bool_col)
    t3_opt = simplify(t3.op())
    assert t3_opt == ops.Project(parent=t, values={"bool_col": t.bool_col})


def test_simplify_subsequent_value_selections():
    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    t2 = t1.select(t1.bool_col, t1.int_col, t1.float_col)
    t2_opt = simplify(t2.op())
    assert t2_opt == ops.Project(
        parent=t,
        values={
            "bool_col": ~t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": t.float_col * 3,
        },
    )

    t3 = t2.select(
        t2.bool_col,
        t2.int_col,
        float_col=t2.float_col * 2,
        another_col=t1.float_col - 1,
    )
    t3_opt = simplify(t3.op())
    assert t3_opt == ops.Project(
        parent=t,
        values={
            "bool_col": ~t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": (t.float_col * 3) * 2,
            "another_col": (t.float_col * 3) - 1,
        },
    )


def test_simplify_subsequent_filters():
    f1 = t.filter(t.bool_col)
    f2 = f1.filter(t.int_col > 0)
    f2_opt = simplify(f2.op())
    assert f2_opt == ops.Filter(t, predicates=[t.bool_col, t.int_col > 0])


def test_simplify_project_filter_project():
    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    t2 = t1.filter(t1.bool_col)
    t3 = t2.filter(t2.int_col > 0)
    t4 = t3.select(t3.bool_col, t3.int_col)

    filt = ops.Filter(parent=t, predicates=[~t.bool_col, t.int_col + 1 > 0]).to_expr()
    proj = ops.Project(
        parent=filt, values={"bool_col": ~filt.bool_col, "int_col": filt.int_col + 1}
    ).to_expr()

    t4_opt = simplify(t4.op())
    assert t4_opt == proj.op()
