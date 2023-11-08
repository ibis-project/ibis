from __future__ import annotations

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
