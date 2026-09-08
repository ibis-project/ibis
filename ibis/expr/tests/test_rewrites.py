from __future__ import annotations

import pytest

import ibis
import ibis.expr.operations as ops
from ibis.expr.operations.reductions import LimitedArrayCollect
from ibis.expr.rewrites import lower_array_collect_slice, simplify

t = ibis.table(
    name="t",
    schema={
        "bool_col": "boolean",
        "int_col": "int64",
        "float_col": "float64",
        "string_col": "string",
    },
)


@pytest.mark.parametrize("stop", [0, 3])
def test_lower_array_collect_prefix_slice(stop):
    """Lower nonnegative literal prefix slices to bounded collection."""
    result = t.int_col.collect()[:stop].op().replace(lower_array_collect_slice)

    assert isinstance(result, LimitedArrayCollect)
    assert result.limit.value == stop


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(t.int_col.collect()[1:3], id="offset"),
        pytest.param(t.int_col.collect()[:-1], id="negative"),
        pytest.param(t.int_col.collect()[:], id="open"),
        pytest.param(t.int_col.collect()[: ibis.null().cast("int64")], id="null"),
        pytest.param(t.int_col.collect()[: t.int_col.max()], id="dynamic"),
        pytest.param(t.int_col.collect().over()[:3], id="windowed-collect"),
    ],
)
def test_preserve_collect_slice(expr):
    """Preserve collection slices that native aggregate bounds cannot express."""
    op = expr.op()

    assert op.replace(lower_array_collect_slice) == op


def test_preserve_slice_around_limited_collect():
    """Do not widen an inner collection bound with an outer slice."""
    op = t.int_col.collect()[:2][:10].op()
    result = op.replace(lower_array_collect_slice)

    assert isinstance(result, ops.ArraySlice)
    assert isinstance(result.arg, LimitedArrayCollect)
    assert result.arg.limit.value == 2


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
