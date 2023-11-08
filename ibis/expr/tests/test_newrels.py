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
    assert f.relations == frozenset([t.op()])


def test_relation_coercion():
    assert ops.Relation.__coerce__(t) == t.op()
    assert ops.Relation.__coerce__(t.op()) == t.op()
    with pytest.raises(TypeError):
        assert ops.Relation.__coerce__("invalid")


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
    assert node.fields == {
        "bool_col": ops.Field(node, "bool_col"),
        "int_col": ops.Field(node, "int_col"),
        "float_col": ops.Field(node, "float_col"),
        "string_col": ops.Field(node, "string_col"),
    }
    assert node.values == {}


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

    assert expected.fields == {
        "1": ops.Field(proj, "1"),
        "float_col": ops.Field(proj, "float_col"),
        "length": ops.Field(proj, "length"),
    }
    assert expected.values == {
        "1": ibis.literal(1).op(),
        "float_col": t.float_col.op(),
        "length": t.string_col.length().op(),
    }


def test_select_windowing_local_reduction():
    t1 = t.select(res=t.int_col.sum())
    assert t1.op() == Project(parent=t, values={"res": t.int_col.sum().over()})


def test_select_windowizing_analytic_function():
    t1 = t.select(res=t.int_col.lag())
    assert t1.op() == Project(parent=t, values={"res": t.int_col.lag().over()})


def test_subquery_integrity_check():
    t = ibis.table(name="t", schema={"a": "int64", "b": "string"})

    msg = "Subquery must have exactly one column, got 2"
    with pytest.raises(IntegrityError, match=msg):
        ops.ScalarSubquery(t)


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
    assert t1.select(t1.bool_col, t.int_col).equals(t2)
    assert t1.select(t.bool_col, t.int_col).equals(t2)
    assert t2.op() == Project(
        parent=t1,
        values={
            "bool_col": t1.bool_col,
            "int_col": t1.int_col,
        },
    )

    t3 = t2.select(t2.bool_col)
    assert t2.select(t1.bool_col).equals(t3)
    assert t2.select(t.bool_col).equals(t3)
    assert t3.op() == Project(
        parent=t2,
        values={
            "bool_col": t2.bool_col,
        },
    )

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
    assert t2.op() == Project(
        parent=t1,
        values={
            "bool_col": t1.bool_col,
            "int_col": t1.int_col,
            "float_col": t1.float_col,
        },
    )

    t3 = t2.select(
        t2.bool_col,
        t2.int_col,
        float_col=t2.float_col * 2,
        another_col=t1.float_col - 1,
    )
    assert t3.op() == Project(
        parent=t2,
        values={
            "bool_col": t2.bool_col,
            "int_col": t2.int_col,
            "float_col": t2.float_col * 2,
            "another_col": t2.float_col - 1,
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

    assert expected.fields == {
        "bool_col": ops.Field(expected, "bool_col"),
        "int_col": ops.Field(expected, "int_col"),
        "float_col": ops.Field(expected, "float_col"),
        "string_col": ops.Field(expected, "string_col"),
    }
    assert expected.values == {
        "bool_col": t.bool_col.op(),
        "int_col": t.int_col.op(),
        "float_col": t.float_col.op(),
        "string_col": t.string_col.op(),
    }


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


def test_where_flattens_predicates():
    t1 = t.filter(t.bool_col & ((t.int_col > 0) & (t.float_col < 0)))
    expected = Filter(
        parent=t,
        predicates=[
            t.bool_col,
            t.int_col > 0,
            t.float_col < 0,
        ],
    )
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


# TODO(kszucs): add test for failing integrity checks
def test_join():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    joined = t1.join(t2, [t1.a == t2.c])
    assert isinstance(joined, ir.JoinExpr)
    assert isinstance(joined.op(), JoinChain)
    assert isinstance(joined.op().to_expr(), ir.JoinExpr)

    result = joined._finish()
    assert isinstance(joined, ir.TableExpr)
    assert isinstance(joined.op(), JoinChain)
    assert isinstance(joined.op().to_expr(), ir.JoinExpr)

    t2_ = joined.op().rest[0].table.to_expr()
    assert result.op() == JoinChain(
        first=t1,
        rest=[
            JoinLink("inner", t2_, [t1.a == t2_.c]),
        ],
        values={
            "a": t1.a,
            "b": t1.b,
            "c": t2_.c,
            "d": t2_.d,
        },
    )


def test_join_unambiguous_select():
    a = ibis.table(name="a", schema={"a_int": "int64", "a_str": "string"})
    b = ibis.table(name="b", schema={"b_int": "int64", "b_str": "string"})

    join = a.join(b, a.a_int == b.b_int)
    expr1 = join["a_int", "b_int"]
    expr2 = join.select("a_int", "b_int")
    assert expr1.equals(expr2)

    b_ = join.op().rest[0].table.to_expr()
    assert expr1.op() == JoinChain(
        first=a,
        rest=[JoinLink("inner", b_, [a.a_int == b_.b_int])],
        values={
            "a_int": a.a_int,
            "b_int": b_.b_int,
        },
    )


def test_join_with_subsequent_projection():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    # a single computed value is pulled to a subsequent projection
    joined = t1.join(t2, [t1.a == t2.c])
    expr = joined.select(t1.a, t1.b, col=t2.c + 1)
    t2_ = joined.op().rest[0].table.to_expr()
    expected = JoinChain(
        first=t1,
        rest=[JoinLink("inner", t2_, [t1.a == t2_.c])],
        values={"a": t1.a, "b": t1.b, "col": t2_.c + 1},
    )
    assert expr.op() == expected

    # multiple computed values
    joined = t1.join(t2, [t1.a == t2.c])
    expr = joined.select(
        t1.a,
        t1.b,
        foo=t2.c + 1,
        bar=t2.c + 2,
        baz=t2.d.name("bar") + "3",
        baz2=(t2.c + t1.a).name("foo"),
    )
    t2_ = joined.op().rest[0].table.to_expr()
    expected = JoinChain(
        first=t1,
        rest=[JoinLink("inner", t2_, [t1.a == t2_.c])],
        values={
            "a": t1.a,
            "b": t1.b,
            "foo": t2_.c + 1,
            "bar": t2_.c + 2,
            "baz": t2_.d.name("bar") + "3",
            "baz2": t2_.c + t1.a,
        },
    )
    assert expr.op() == expected


def test_join_with_subsequent_projection_colliding_names():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(
        name="t2", schema={"a": "int64", "b": "string", "c": "float", "d": "string"}
    )

    joined = t1.join(t2, [t1.a == t2.a])
    expr = joined.select(
        t1.a,
        t1.b,
        foo=t2.a + 1,
        bar=t1.a + t2.a,
    )
    t2_ = joined.op().rest[0].table.to_expr()
    expected = JoinChain(
        first=t1,
        rest=[JoinLink("inner", t2_, [t1.a == t2_.a])],
        values={
            "a": t1.a,
            "b": t1.b,
            "foo": t2_.a + 1,
            "bar": t1.a + t2_.a,
        },
    )
    assert expr.op() == expected


def test_chained_join():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})

    joined = a.join(b, [a.a == b.c]).join(c, [a.a == c.e])
    result = joined._finish()

    b_ = joined.op().rest[0].table.to_expr()
    c_ = joined.op().rest[1].table.to_expr()
    assert result.op() == JoinChain(
        first=a,
        rest=[
            JoinLink("inner", b_, [a.a == b_.c]),
            JoinLink("inner", c_, [a.a == c_.e]),
        ],
        values={
            "a": a.a,
            "b": a.b,
            "c": b_.c,
            "d": b_.d,
            "e": c_.e,
            "f": c_.f,
        },
    )

    joined = a.join(b, [a.a == b.c]).join(c, [b.c == c.e])
    result = joined.select(a.a, b.d, c.f)

    b_ = joined.op().rest[0].table.to_expr()
    c_ = joined.op().rest[1].table.to_expr()
    assert result.op() == JoinChain(
        first=a,
        rest=[
            JoinLink("inner", b_, [a.a == b_.c]),
            JoinLink("inner", c_, [b_.c == c_.e]),
        ],
        values={
            "a": a.a,
            "d": b_.d,
            "f": c_.f,
        },
    )


def test_chained_join_referencing_intermediate_table():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})

    ab = a.join(b, [a.a == b.c])
    assert isinstance(ab, ir.JoinExpr)

    # assert ab.a.op() == Field(ab, "a")
    abc = ab.join(c, [ab.a == c.e])
    assert isinstance(abc, ir.JoinExpr)

    result = abc._finish()

    b_ = abc.op().rest[0].table.to_expr()
    c_ = abc.op().rest[1].table.to_expr()
    assert result.op() == JoinChain(
        first=a,
        rest=[
            JoinLink("inner", b_, [a.a == b_.c]),
            JoinLink("inner", c_, [a.a == c_.e]),
        ],
        values={"a": a.a, "b": a.b, "c": b_.c, "d": b_.d, "e": c_.e, "f": c_.f},
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

    table2_ = j1.op().rest[0].table.to_expr()
    expected = ops.JoinChain(
        first=filtered,
        rest=[
            ops.JoinLink("left", table2_, [filtered.foo_id == table2_.foo_id]),
        ],
        values={
            "c": filtered.c,
            "f": filtered.f,
            "foo_id": filtered.foo_id,
            "bar_id": filtered.bar_id,
            "foo_id_right": table2_.foo_id,
            "value1": table2_.value1,
            "value3": table2_.value3,
        },
    )
    assert j1.op() == expected

    j2 = j1.inner_join(table3, filtered["bar_id"] == table3["bar_id"])

    table2_ = j2.op().rest[0].table.to_expr()
    table3_ = j2.op().rest[1].table.to_expr()
    expected = ops.JoinChain(
        first=filtered,
        rest=[
            ops.JoinLink("left", table2_, [filtered.foo_id == table2_.foo_id]),
            ops.JoinLink("inner", table3_, [filtered.bar_id == table3_.bar_id]),
        ],
        values={
            "c": filtered.c,
            "f": filtered.f,
            "foo_id": filtered.foo_id,
            "bar_id": filtered.bar_id,
            "foo_id_right": table2_.foo_id,
            "value1": table2_.value1,
            "value3": table2_.value3,
            "bar_id_right": table3_.bar_id,
            "value2": table3_.value2,
        },
    )
    assert j2.op() == expected

    # Project out the desired fields
    view = j2[[filtered, table2["value1"], table3["value2"]]]
    expected = ops.JoinChain(
        first=filtered,
        rest=[
            ops.JoinLink("left", table2_, [filtered.foo_id == table2_.foo_id]),
            ops.JoinLink("inner", table3_, [filtered.bar_id == table3_.bar_id]),
        ],
        values={
            "c": filtered.c,
            "f": filtered.f,
            "foo_id": filtered.foo_id,
            "bar_id": filtered.bar_id,
            "value1": table2_.value1,
            "value2": table3_.value2,
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


def test_self_join():
    t0 = ibis.table(schema=ibis.schema(dict(key="int")), name="leaf")
    t1 = t0.filter(ibis.literal(True))
    t2 = t1[["key"]]

    t3 = t2.join(t2, ["key"])
    t2_ = t3.op().rest[0].table.to_expr()
    expected = ops.JoinChain(
        first=t2,
        rest=[
            ops.JoinLink("inner", t2_, [t2.key == t2_.key]),
        ],
        values={"key": t2.key, "key_right": t2_.key},
    )
    assert t3.op() == expected

    t4 = t3.join(t3, ["key"])
    t3_ = t4.op().rest[1].table.to_expr()

    expected = ops.JoinChain(
        first=t2,
        rest=[
            ops.JoinLink("inner", t2_, [t2.key == t2_.key]),
            ops.JoinLink("inner", t3_, [t2.key == t3_.key]),
        ],
        values={
            "key": t2.key,
            "key_right": t2_.key,
            "key_right_right": t3_.key_right,
        },
    )
    assert t4.op() == expected


def test_self_join_view():
    t = ibis.memtable({"x": [1, 2], "y": [2, 1], "z": ["a", "b"]})
    t_view = t.view()
    expr = t.join(t_view, t.x == t_view.y).select("x", "y", "z", "z_right")

    t_view_ = expr.op().rest[0].table.to_expr()
    expected = ops.JoinChain(
        first=t,
        rest=[
            ops.JoinLink("inner", t_view_, [t.x == t_view_.y]),
        ],
        values={"x": t.x, "y": t.y, "z": t.z, "z_right": t_view_.z},
    )
    assert expr.op() == expected


def test_self_join_with_view_projection():
    t1 = ibis.memtable({"x": [1, 2], "y": [2, 1], "z": ["a", "b"]})
    t2 = t1.view()
    expr = t1.inner_join(t2, ["x"])[[t1]]

    t2_ = expr.op().rest[0].table.to_expr()
    expected = ops.JoinChain(
        first=t1,
        rest=[
            ops.JoinLink("inner", t2_, [t1.x == t2_.x]),
        ],
        values={"x": t1.x, "y": t1.y, "z": t1.z},
    )
    assert expr.op() == expected


def test_joining_same_table_twice():
    left = ibis.table(name="left", schema={"time1": int, "value": float, "a": str})
    right = ibis.table(name="right", schema={"time2": int, "value2": float, "b": str})

    joined = left.inner_join(right, left.a == right.b).inner_join(
        right, left.value == right.value2
    )

    right_ = joined.op().rest[0].table.to_expr()
    right__ = joined.op().rest[1].table.to_expr()
    expected = ops.JoinChain(
        first=left,
        rest=[
            ops.JoinLink("inner", right_, [left.a == right_.b]),
            ops.JoinLink("inner", right__, [left.value == right__.value2]),
        ],
        values={
            "time1": left.time1,
            "value": left.value,
            "a": left.a,
            "time2": right_.time2,
            "value2": right_.value2,
            "b": right_.b,
            "time2_right": right__.time2,
            "value2_right": right__.value2,
            "b_right": right__.b,
        },
    )
    assert joined.op() == expected


def test_join_chain_gets_reused_and_continued_after_a_select():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})

    ab = a.join(b, [a.a == b.c])
    abc = ab[a.b, b.d].join(c, [a.a == c.e])

    b_ = abc.op().rest[0].table.to_expr()
    c_ = abc.op().rest[1].table.to_expr()
    expected = ops.JoinChain(
        first=a,
        rest=[
            ops.JoinLink("inner", b_, [a.a == b_.c]),
            ops.JoinLink("inner", c_, [a.a == c_.e]),
        ],
        values={
            "b": a.b,
            "d": b_.d,
            "e": c_.e,
            "f": c_.f,
        },
    )
    assert abc.op() == expected
    assert abc._finish().op() == expected


def test_self_join_extensive():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})

    aa = a.join(a, [a.a == a.a])
    aa_ = a.join(a, "a")
    aa__ = a.join(a, [("a", "a")])
    for join in [aa, aa_, aa__]:
        a1 = join.op().rest[0].table.to_expr()
        expected = ops.JoinChain(
            first=a,
            rest=[
                ops.JoinLink("inner", a1, [a.a == a1.a]),
            ],
            values={
                "a": a.a,
                "b": a.b,
                "a_right": a1.a,
                "b_right": a1.b,
            },
        )
        assert join.op() == expected

    aaa = a.join(a, [a.a == a.a]).join(a, [a.a == a.a])
    a0 = a
    a1 = aaa.op().rest[0].table.to_expr()
    a2 = aaa.op().rest[1].table.to_expr()
    expected = ops.JoinChain(
        first=a0,
        rest=[
            ops.JoinLink("inner", a1, [a0.a == a1.a]),
            ops.JoinLink("inner", a2, [a0.a == a2.a]),
        ],
        values={
            "a": a0.a,
            "b": a0.b,
            "a_right": a1.a,
            "b_right": a1.b,
        },
    )

    aaa = aa.join(a, [aa.a == a.a])
    aaa_ = aa.join(a, "a")
    aaa__ = aa.join(a, [("a", "a")])
    for join in [aaa, aaa_, aaa__]:
        a1 = join.op().rest[0].table.to_expr()
        a2 = join.op().rest[1].table.to_expr()
        expected = ops.JoinChain(
            first=a,
            rest=[
                ops.JoinLink("inner", a1, [a.a == a1.a]),
                ops.JoinLink("inner", a2, [a.a == a2.a]),
            ],
            values={
                "a": a.a,
                "b": a.b,
                "a_right": a1.a,
                "b_right": a1.b,
            },
        )
        assert join.op() == expected


def test_self_join_with_intermediate_selection():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})

    join = a[["b", "a"]].join(a, [a.a == a.a])
    a0 = a[["b", "a"]]
    a1 = join.op().rest[0].table.to_expr()
    expected = ops.JoinChain(
        first=a0,
        rest=[
            ops.JoinLink("inner", a1, [a0.a == a1.a]),
        ],
        values={
            "b": a0.b,
            "a": a0.a,
            "a_right": a1.a,
            "b_right": a1.b,
        },
    )
    assert join.op() == expected

    aa_ = a.join(a, [a.a == a.a])["a", "b_right"]
    aaa_ = aa_.join(a, [aa_.a == a.a])
    a0 = a
    a1 = aaa_.op().rest[0].table.to_expr()
    a2 = aaa_.op().rest[1].table.to_expr()
    expected = ops.JoinChain(
        first=a0,
        rest=[
            ops.JoinLink("inner", a1, [a0.a == a1.a]),
            ops.JoinLink("inner", a2, [a0.a == a2.a]),
        ],
        values={
            "a": a0.a,
            "b_right": a1.b,
            "a_right": a2.a,
            "b": a2.b,
        },
    )
    assert aaa_.op() == expected

    # TODO(kszucs): this use case could be supported if `_get_column` gets
    # overridden to return underlying column reference, but that would mean
    # that `aa.a` returns with `a.a` instead of `aa.a` which breaks other
    # things
    # aa = a.join(a, [a.a == a.a])
    # aaa = aa["a", "b_right"].join(a, [aa.a == a.a])
    # a0 = a
    # a1 = aaa.op().rest[0].table.to_expr()
    # a2 = aaa.op().rest[1].table.to_expr()
    # expected = ops.JoinChain(
    #     first=a0,
    #     rest=[
    #         ops.JoinLink("inner", a1, [a0.a == a1.a]),
    #         ops.JoinLink("inner", a2, [a0.a == a2.a]),
    #     ],
    #     values={
    #         "a": a0.a,
    #         "b_right": a1.b,
    #         "a_right": a2.a,
    #         "b": a2.b,
    #     },
    # )
    # assert aaa.op() == expected


def test_name_collisions_raise():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"a": "int64", "b": "string"})
    c = ibis.table(name="c", schema={"a": "int64", "b": "string"})

    ab = a.join(b, [a.a == b.a])
    filt = ab.filter(ab.a < 1)
    expected = ops.Filter(
        parent=ab,
        predicates=[
            ops.Less(ops.Field(ab, "a"), 1),
        ],
    )
    assert filt.op() == expected

    abc = a.join(b, [a.a == b.a]).join(c, [a.a == c.a])
    with pytest.raises(IntegrityError):
        abc.filter(abc.a < 1)
