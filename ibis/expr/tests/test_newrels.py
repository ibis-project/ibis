from __future__ import annotations

import pytest

import ibis
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import _
from ibis.common.annotations import ValidationError
from ibis.common.exceptions import IbisInputError, IntegrityError
from ibis.expr.operations import (
    Aggregate,
    Field,
    Filter,
    JoinChain,
    JoinLink,
    Project,
    UnboundTable,
)
from ibis.expr.schema import Schema

# TODO(kszucs):
# def test_relation_coercion()
# def test_where_flattens_predicates()

t = ibis.table(
    name="t",
    schema={
        "bool_col": "boolean",
        "int_col": "int64",
        "float_col": "float64",
        "string_col": "string",
    },
)


def test_field():
    f = Field(t, "bool_col")
    assert f.rel == t.op()
    assert f.name == "bool_col"
    assert f.shape == ds.columnar
    assert f.dtype == dt.boolean
    assert f.to_expr().equals(t.bool_col)


def test_unbound_table():
    node = t.op()
    assert isinstance(t, ir.TableExpr)
    assert isinstance(node, UnboundTable)
    assert node.name == "t"
    assert node.schema == Schema(
        {
            "bool_col": dt.boolean,
            "int_col": dt.int64,
            "float_col": dt.float64,
            "string_col": dt.string,
        }
    )


def test_select_fields():
    proj = t.select("int_col")
    expected = Project(parent=t, values={"int_col": t.int_col})
    assert proj.op() == expected
    assert proj.op().schema == Schema({"int_col": dt.int64})

    proj = t.select(myint=t.int_col)
    expected = Project(parent=t, values={"myint": t.int_col})
    assert proj.op() == expected
    assert proj.op().schema == Schema({"myint": dt.int64})

    proj = t.select(t.int_col, myint=t.int_col)
    expected = Project(parent=t, values={"int_col": t.int_col, "myint": t.int_col})
    assert proj.op() == expected
    assert proj.op().schema == Schema({"int_col": dt.int64, "myint": dt.int64})

    proj = t.select(_.int_col, myint=_.int_col)
    expected = Project(parent=t, values={"int_col": t.int_col, "myint": t.int_col})
    assert proj.op() == expected


def test_select_values():
    proj = t.select((1 + t.int_col).name("incremented"))
    expected = Project(parent=t, values={"incremented": (1 + t.int_col)})
    assert proj.op() == expected
    assert proj.op().schema == Schema({"incremented": dt.int64})

    proj = t.select(ibis.literal(1), "float_col", length=t.string_col.length())
    expected = Project(
        parent=t,
        values={"1": 1, "float_col": t.float_col, "length": t.string_col.length()},
    )
    assert proj.op() == expected
    assert proj.op().schema == Schema(
        {"1": dt.int8, "float_col": dt.float64, "length": dt.int32}
    )


def test_select_windowing_local_reduction():
    t1 = t.select(res=t.int_col.sum())
    assert t1.op() == Project(parent=t, values={"res": t.int_col.sum().over()})


def test_select_windowizing_analytic_function():
    t1 = t.select(res=t.int_col.lag())
    assert t1.op() == Project(parent=t, values={"res": t.int_col.lag().over()})


def test_select_turns_scalar_reduction_into_subquery():
    arr = ibis.literal([1, 2, 3])
    res = arr.unnest().sum()
    t1 = t.select(res)
    subquery = ops.ScalarSubquery(res.as_table())
    expected = Project(parent=t, values={"Sum((1, 2, 3))": subquery})
    assert t1.op() == expected


def test_select_scalar_foreign_scalar_reduction_into_subquery():
    t1 = t.filter(t.bool_col)
    t2 = t.select(summary=t1.int_col.sum())
    subquery = ops.ScalarSubquery(t1.int_col.sum().as_table())
    expected = Project(parent=t, values={"summary": subquery})
    assert t2.op() == expected


def test_select_turns_value_with_multiple_parents_into_subquery():
    v = ibis.table(name="v", schema={"a": "int64", "b": "string"})
    v_filt = v.filter(v.a == t.int_col)

    t1 = t.select(t.int_col, max=v_filt.a.max())
    subquery = ops.ScalarSubquery(v_filt.a.max().as_table())
    expected = Project(parent=t, values={"int_col": t.int_col, "max": subquery})
    assert t1.op() == expected


def test_mutate():
    proj = t.select(t, other=t.int_col + 1)
    expected = Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
            "float_col": t.float_col,
            "string_col": t.string_col,
            "other": t.int_col + 1,
        },
    )
    assert proj.op() == expected


def test_mutate_overwrites_existing_column():
    t = ibis.table(dict(a="string", b="string"))

    mut = t.mutate(a=42)
    assert mut.op() == Project(parent=t, values={"a": ibis.literal(42), "b": t.b})

    sel = mut.select("a")
    assert sel.op() == Project(parent=mut, values={"a": mut.a})


def test_select_full_reprojection():
    t1 = t.select(t)
    assert t1.op() == Project(
        t,
        {
            "bool_col": t.bool_col,
            "int_col": t.int_col,
            "float_col": t.float_col,
            "string_col": t.string_col,
        },
    )

    t1_opt = t1.optimize()
    assert t1_opt.op() == t.op()


def test_subsequent_selections_with_field_names():
    t1 = t.select("bool_col", "int_col", "float_col")
    assert t1.op() == Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
            "float_col": t.float_col,
        },
    )
    t2 = t1.select("bool_col", "int_col")
    assert t2.op() == Project(
        parent=t1,
        values={
            "bool_col": t1.bool_col,
            "int_col": t1.int_col,
        },
    )
    t3 = t2.select("bool_col")
    assert t3.op() == Project(
        parent=t2,
        values={
            "bool_col": t2.bool_col,
        },
    )

    t2_opt = t2.optimize()
    assert t2_opt.op() == Project(
        parent=t, values={"bool_col": t.bool_col, "int_col": t.int_col}
    )

    t3_opt = t3.optimize()
    assert t3_opt.op() == Project(parent=t, values={"bool_col": t.bool_col})


def test_subsequent_selections_field_dereferencing():
    t1 = t.select(t.bool_col, t.int_col, t.float_col)
    assert t1.op() == Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
            "float_col": t.float_col,
        },
    )

    t2 = t1.select(t1.bool_col, t1.int_col)
    t2_opt = t2.optimize()
    assert t1.select(t1.bool_col, t.int_col).equals(t2)
    assert t1.select(t.bool_col, t.int_col).equals(t2)
    assert t2.op() == Project(
        parent=t1,
        values={
            "bool_col": t1.bool_col,
            "int_col": t1.int_col,
        },
    )
    assert t2_opt.op() == Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
        },
    )

    t3 = t2.select(t2.bool_col)
    t3_opt = t3.optimize()
    assert t2.select(t1.bool_col).equals(t3)
    assert t2.select(t.bool_col).equals(t3)
    assert t3.op() == Project(
        parent=t2,
        values={
            "bool_col": t2.bool_col,
        },
    )
    assert t3_opt.op() == Project(parent=t, values={"bool_col": t.bool_col})

    u1 = t.select(t.bool_col, t.int_col, t.float_col)
    assert u1.op() == Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
            "float_col": t.float_col,
        },
    )

    u2 = u1.select(u1.bool_col, u1.int_col, u1.float_col)
    assert u1.select(t.bool_col, u1.int_col, u1.float_col).equals(u2)
    assert u1.select(t.bool_col, t.int_col, t.float_col).equals(u2)
    assert u2.op() == Project(
        parent=u1,
        values={
            "bool_col": u1.bool_col,
            "int_col": u1.int_col,
            "float_col": u1.float_col,
        },
    )

    u3 = u2.select(u2.bool_col, u2.int_col, u2.float_col)
    assert u2.select(u2.bool_col, u1.int_col, u2.float_col).equals(u3)
    assert u2.select(u2.bool_col, u1.int_col, t.float_col).equals(u3)
    assert u3.op() == Project(
        parent=u2,
        values={
            "bool_col": u2.bool_col,
            "int_col": u2.int_col,
            "float_col": u2.float_col,
        },
    )


def test_subsequent_selections_value_dereferencing():
    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    assert t1.op() == Project(
        parent=t,
        values={
            "bool_col": ~t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": t.float_col * 3,
        },
    )

    t2 = t1.select(t1.bool_col, t1.int_col, t1.float_col)
    t2_opt = t2.optimize()
    assert t2.op() == Project(
        parent=t1,
        values={
            "bool_col": t1.bool_col,
            "int_col": t1.int_col,
            "float_col": t1.float_col,
        },
    )
    assert t2_opt.op() == Project(
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

    t3_opt = t3.optimize()
    assert t3.op() == Project(
        parent=t2,
        values={
            "bool_col": t2.bool_col,
            "int_col": t2.int_col,
            "float_col": t2.float_col * 2,
            "another_col": t2.float_col - 1,
        },
    )
    assert t3_opt.op() == Project(
        parent=t,
        values={
            "bool_col": ~t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": (t.float_col * 3) * 2,
            "another_col": (t.float_col * 3) - 1,
        },
    )


def test_where():
    filt = t.filter(t.bool_col)
    expected = Filter(parent=t, predicates=[t.bool_col])
    assert filt.op() == expected

    filt = t.filter(t.bool_col, t.int_col > 0)
    expected = Filter(parent=t, predicates=[t.bool_col, t.int_col > 0])
    assert filt.op() == expected

    filt = t.filter(_.bool_col)
    expected = Filter(parent=t, predicates=[t.bool_col])
    assert filt.op() == expected


def test_where_raies_for_empty_predicate_list():
    t = ibis.table(dict(a="string"))
    with pytest.raises(IbisInputError):
        t.filter()


def test_where_after_select():
    t1 = t.select(t.bool_col)
    t2 = t1.filter(t.bool_col)
    expected = Filter(parent=t1, predicates=[t1.bool_col])
    assert t2.op() == expected

    t1 = t.select(int_col=t.bool_col)
    t2 = t1.filter(t.bool_col)
    expected = Filter(parent=t1, predicates=[t1.int_col])
    assert t2.op() == expected


def test_where_with_reduction():
    with pytest.raises(IntegrityError):
        Filter(t, predicates=[t.int_col.sum() > 1])

    t1 = t.filter(t.int_col.sum() > 0)
    subquery = ops.ScalarSubquery(t.int_col.sum().as_table())
    expected = Filter(parent=t, predicates=[ops.Greater(subquery, 0)])
    assert t1.op() == expected


def test_project_filter_sort():
    expr = t.select(t.bool_col, t.int_col).filter(t.bool_col).order_by(t.int_col)
    expected = ops.Sort(
        parent=(
            filt := ops.Filter(
                parent=(
                    proj := ops.Project(
                        parent=t,
                        values={
                            "bool_col": t.bool_col,
                            "int_col": t.int_col,
                        },
                    )
                ),
                predicates=[ops.Field(proj, "bool_col")],
            )
        ),
        keys=[ops.SortKey(ops.Field(filt, "int_col"), ascending=True)],
    )
    assert expr.op() == expected


def test_subsequent_filter():
    f1 = t.filter(t.bool_col)
    f2 = f1.filter(t.int_col > 0)
    expected = Filter(f1, predicates=[f1.int_col > 0])
    assert f2.op() == expected

    f2_opt = f2.optimize()
    assert f2_opt.op() == Filter(t, predicates=[t.bool_col, t.int_col > 0])


def test_project_before_and_after_filter():
    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    assert t1.op() == Project(
        parent=t,
        values={
            "bool_col": ~t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": t.float_col * 3,
        },
    )

    t2 = t1.filter(t1.bool_col)
    assert t2.op() == Filter(parent=t1, predicates=[t1.bool_col])

    t3 = t2.filter(t2.int_col > 0)
    assert t3.op() == Filter(parent=t2, predicates=[t2.int_col > 0])

    t3_ = t2.filter(t1.int_col > 0)
    assert t3_.op() == Filter(parent=t2, predicates=[t2.int_col > 0])

    t4 = t3.select(t3.bool_col, t3.int_col)
    assert t4.op() == Project(
        parent=t3,
        values={
            "bool_col": t3.bool_col,
            "int_col": t3.int_col,
        },
    )

    t2_opt = t2.optimize(enable_reordering=False)
    assert t2_opt.op() == Filter(parent=t1, predicates=[t1.bool_col])

    t3_opt = t3.optimize(enable_reordering=False)
    assert t3_opt.op() == Filter(parent=t1, predicates=[t1.bool_col, t1.int_col > 0])

    t4_opt = t4.optimize(enable_reordering=False)
    assert t4_opt.op() == Project(
        parent=t3_opt, values={"bool_col": t3_opt.bool_col, "int_col": t3_opt.int_col}
    )


# TODO(kszucs): add test for failing integrity checks
def test_join():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    joined = t1.join(t2, [t1.a == t2.c])
    assert isinstance(joined, ir.JoinExpr)
    assert isinstance(joined.op(), JoinChain)
    assert isinstance(joined.op().to_expr(), ir.JoinExpr)

    result = joined.finish()
    assert isinstance(joined, ir.TableExpr)
    assert isinstance(joined.op(), JoinChain)
    assert isinstance(joined.op().to_expr(), ir.JoinExpr)

    assert result.op() == JoinChain(
        first=t1,
        rest=[
            JoinLink("inner", t2, [t1.a == t2.c]),
        ],
        fields={
            "a": t1.a,
            "b": t1.b,
            "c": t2.c,
            "d": t2.d,
        },
    )


def test_join_unambiguous_select():
    a = ibis.table(name="a", schema={"a_int": "int64", "a_str": "string"})
    b = ibis.table(name="b", schema={"b_int": "int64", "b_str": "string"})

    join = a.join(b, a.a_int == b.b_int)
    expr1 = join["a_int", "b_int"]
    expr2 = join.select("a_int", "b_int")
    assert expr1.equals(expr2)
    assert expr1.op() == JoinChain(
        first=a,
        rest=[JoinLink("inner", b, [a.a_int == b.b_int])],
        fields={
            "a_int": a.a_int,
            "b_int": b.b_int,
        },
    )


def test_join_with_subsequent_value_projection():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    joined = t1.join(t2, [t1.a == t2.c])
    expr = joined.select(t1.a, t1.b, col=t2.c + 1)
    assert expr.op() == JoinChain(
        first=t1,
        rest=[JoinLink("inner", t2, [t1.a == t2.c])],
        fields={"a": t1.a, "b": t1.b, "col": t2.c + 1},
    )


def test_chained_join():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})

    joined = a.join(b, [a.a == b.c]).join(c, [a.a == c.e])
    result = joined.finish()
    assert result.op() == JoinChain(
        first=a,
        rest=[
            JoinLink("inner", b, [a.a == b.c]),
            JoinLink("inner", c, [a.a == c.e]),
        ],
        fields={
            "a": a.a,
            "b": a.b,
            "c": b.c,
            "d": b.d,
            "e": c.e,
            "f": c.f,
        },
    )

    joined = a.join(b, [a.a == b.c]).join(c, [b.c == c.e])
    result = joined.select(a.a, b.d, c.f)
    assert result.op() == JoinChain(
        first=a,
        rest=[
            JoinLink("inner", b, [a.a == b.c]),
            JoinLink("inner", c, [b.c == c.e]),
        ],
        fields={
            "a": a.a,
            "d": b.d,
            "f": c.f,
        },
    )


def test_chained_join_referencing_intermediate_table():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})

    ab = a.join(b, [a.a == b.c])
    assert isinstance(ab, ir.JoinExpr)

    assert ab.a.op() == Field(ab, "a")
    abc = ab.join(c, [ab.a == c.e])
    assert isinstance(abc, ir.JoinExpr)

    result = abc.finish()
    assert result.op() == JoinChain(
        first=a,
        rest=[JoinLink("inner", b, [a.a == b.c]), JoinLink("inner", c, [a.a == c.e])],
        fields={"a": a.a, "b": a.b, "c": b.c, "d": b.d, "e": c.e, "f": c.f},
    )


def test_join_predicate_dereferencing():
    # See #790, predicate pushdown in joins not supported

    # Star schema with fact table
    table = ibis.table({"c": int, "f": float, "foo_id": str, "bar_id": str})
    table2 = ibis.table({"foo_id": str, "value1": float, "value3": float})
    table3 = ibis.table({"bar_id": str, "value2": float})

    filtered = table[table["f"] > 0]

    # dereference table.foo_id to filtered.foo_id
    j1 = filtered.left_join(table2, table["foo_id"] == table2["foo_id"])
    expected = ops.JoinChain(
        first=filtered,
        rest=[
            ops.JoinLink("left", table2, [filtered.foo_id == table2.foo_id]),
        ],
        fields={
            "c": filtered.c,
            "f": filtered.f,
            "foo_id": filtered.foo_id,
            "bar_id": filtered.bar_id,
            "foo_id_right": table2.foo_id,
            "value1": table2.value1,
            "value3": table2.value3,
        },
    )
    assert j1.op() == expected

    j2 = j1.inner_join(table3, filtered["bar_id"] == table3["bar_id"])
    expected = ops.JoinChain(
        first=filtered,
        rest=[
            ops.JoinLink("left", table2, [filtered.foo_id == table2.foo_id]),
            ops.JoinLink("inner", table3, [filtered.bar_id == table3.bar_id]),
        ],
        fields={
            "c": filtered.c,
            "f": filtered.f,
            "foo_id": filtered.foo_id,
            "bar_id": filtered.bar_id,
            "foo_id_right": table2.foo_id,
            "value1": table2.value1,
            "value3": table2.value3,
            "bar_id_right": table3.bar_id,
            "value2": table3.value2,
        },
    )
    assert j2.op() == expected

    # Project out the desired fields
    view = j2[[filtered, table2["value1"], table3["value2"]]]
    expected = ops.JoinChain(
        first=filtered,
        rest=[
            ops.JoinLink("left", table2, [filtered.foo_id == table2.foo_id]),
            ops.JoinLink("inner", table3, [filtered.bar_id == table3.bar_id]),
        ],
        fields={
            "c": filtered.c,
            "f": filtered.f,
            "foo_id": filtered.foo_id,
            "bar_id": filtered.bar_id,
            "value1": table2.value1,
            "value2": table3.value2,
        },
    )
    assert view.op() == expected


def test_aggregate():
    agg = t.aggregate(by=[t.bool_col], metrics=[t.int_col.sum()])
    expected = Aggregate(
        parent=t,
        groups={
            "bool_col": t.bool_col,
        },
        metrics={
            "Sum(int_col)": t.int_col.sum(),
        },
    )
    assert agg.op() == expected


def test_aggregate_having():
    table = ibis.table(name="table", schema={"g": "string", "f": "double"})

    metrics = [table.f.sum().name("total")]
    by = ["g"]

    expr = table.aggregate(metrics, by=by, having=(table.f.sum() > 0).name("cond"))
    expected = table.aggregate(metrics, by=by).filter(_.total > 0)
    assert expr.equals(expected)

    with pytest.raises(ValidationError):
        # non boolean
        table.aggregate(metrics, by=by, having=table.f.sum())

    with pytest.raises(IntegrityError):
        # non scalar
        table.aggregate(metrics, by=by, having=table.f > 2)


def test_select_with_uncorrelated_scalar_subquery():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    # Create a subquery
    t2_filt = t2.filter(t2.d == "value")

    # Non-reduction won't be turned into a subquery
    with pytest.raises(IntegrityError):
        t1.select(t2_filt.c)

    # Construct the projection using the subquery
    sub = t1.select(t1.a, summary=t2_filt.c.sum())
    expected = Project(
        parent=t1,
        values={
            "a": t1.a,
            "summary": ops.ScalarSubquery(t2_filt.c.sum().as_table()),
        },
    )
    assert sub.op() == expected


def test_select_with_reduction_turns_into_window_function():
    # Define your tables
    employees = ibis.table(
        name="employees", schema={"name": "string", "salary": "double"}
    )

    # Use the subquery in a select operation
    expr = employees.select(employees.name, average_salary=employees.salary.mean())
    expected = Project(
        parent=employees,
        values={
            "name": employees.name,
            "average_salary": employees.salary.mean().over(),
        },
    )
    assert expr.op() == expected


def test_select_with_correlated_scalar_subquery():
    # Define your tables
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    # Create a subquery
    filt = t2.filter(t2.d == t1.b)
    summary = filt.c.sum().name("summary")

    # Use the subquery in a select operation
    expr = t1.select(t1.a, summary)
    expected = Project(
        parent=t1,
        values={
            "a": t1.a,
            "summary": ops.ScalarSubquery(filt.c.sum().as_table()),
        },
    )
    assert expr.op() == expected


def test_aggregate_field_dereferencing():
    t = ibis.table(
        {
            "l_orderkey": "int32",
            "l_partkey": "int32",
            "l_suppkey": "int32",
            "l_linenumber": "int32",
            "l_quantity": "decimal(15, 2)",
            "l_extendedprice": "decimal(15, 2)",
            "l_discount": "decimal(15, 2)",
            "l_tax": "decimal(15, 2)",
            "l_returnflag": "string",
            "l_linestatus": "string",
            "l_shipdate": "date",
            "l_commitdate": "date",
            "l_receiptdate": "date",
            "l_shipinstruct": "string",
            "l_shipmode": "string",
            "l_comment": "string",
        }
    )

    f = t.filter(t.l_shipdate <= ibis.date("1998-09-01"))
    assert f.op() == Filter(
        parent=t, predicates=[t.l_shipdate <= ibis.date("1998-09-01")]
    )

    discount_price = t.l_extendedprice * (1 - t.l_discount)
    charge = discount_price * (1 + t.l_tax)
    a = f.group_by(["l_returnflag", "l_linestatus"]).aggregate(
        sum_qty=t.l_quantity.sum(),
        sum_base_price=t.l_extendedprice.sum(),
        sum_disc_price=discount_price.sum(),
        sum_charge=charge.sum(),
        avg_qty=t.l_quantity.mean(),
        avg_price=t.l_extendedprice.mean(),
        avg_disc=t.l_discount.mean(),
        count_order=f.count(),  # note that this is f.count() not t.count()
    )

    discount_price_ = f.l_extendedprice * (1 - f.l_discount)
    charge_ = discount_price_ * (1 + f.l_tax)
    assert a.op() == Aggregate(
        parent=f,
        groups={
            "l_returnflag": f.l_returnflag,
            "l_linestatus": f.l_linestatus,
        },
        metrics={
            "sum_qty": f.l_quantity.sum(),
            "sum_base_price": f.l_extendedprice.sum(),
            "sum_disc_price": discount_price_.sum(),
            "sum_charge": charge_.sum(),
            "avg_qty": f.l_quantity.mean(),
            "avg_price": f.l_extendedprice.mean(),
            "avg_disc": f.l_discount.mean(),
            "count_order": f.count(),
        },
    )

    s = a.order_by(["l_returnflag", "l_linestatus"])
    assert s.op() == ops.Sort(
        parent=a,
        keys=[a.l_returnflag, a.l_linestatus],
    )


def test_sequelize():
    expr = (
        t.select(t.bool_col, t.int_col, incremented=t.int_col + 1)
        .filter(_.incremented < 5)
        .order_by(t.int_col + 1)
    )
    selection = expr.sequelize()
    assert isinstance(selection.to_expr(), ir.Expr)


def test_isin_subquery():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    t2_filt = t2.filter(t2.d == "value")

    expr = t1.filter(t1.a.isin(t2_filt.c))
    subquery = Project(t2_filt, values={"c": t2_filt.c})
    expected = Filter(parent=t1, predicates=[ops.InSubquery(rel=subquery, needle=t1.a)])
    assert expr.op() == expected


def test_filter_condition_referencing_agg_without_groupby_turns_it_into_a_subquery():
    r1 = ibis.table(
        name="r3", schema={"name": str, "key": str, "int_col": int, "float_col": float}
    )
    r2 = r1.filter(r1.name == "GERMANY")
    r3 = r2.aggregate(by=[r2.key], value=(r2.float_col * r2.int_col).sum())
    r4 = r2.aggregate(total=(r2.float_col * r2.int_col).sum())
    r5 = r3.filter(r3.value > r4.total * 0.0001)

    total = (r2.float_col * r2.int_col).sum()
    subquery = ops.ScalarSubquery(
        ops.Aggregate(r2, groups={}, metrics={total.get_name(): total})
    ).to_expr()
    expected = Filter(parent=r3, predicates=[r3.value > subquery * 0.0001])

    assert r5.op() == expected
