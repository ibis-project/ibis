from __future__ import annotations

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from ibis.expr.newrels import (
    Field,
    Filter,
    Join,
    Project,
    Sort,
    TableExpr,
    UnboundTable,
)
from ibis.expr.schema import Schema
from ibis import _

# TODO(kszucs):
# def test_relation_coercion()
# def test_where_flattens_predicates()

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
    assert f.to_expr().equals(t.bool_col)


def test_unbound_table():
    node = t.op()
    assert isinstance(t, TableExpr)
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


def test_select_relation():
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


def test_select_values():
    proj = t.select((1 + t.int_col).name("incremented"))
    expected = Project(parent=t, values={"incremented": (1 + t.int_col)})
    assert proj.op() == expected
    assert proj.op().schema == Schema({"incremented": dt.int64})

    proj = t.select(1, "float_col", length=t.string_col.length())
    expected = Project(
        parent=t,
        values={"1": 1, "float_col": t.float_col, "length": t.string_col.length()},
    )
    assert proj.op() == expected
    assert proj.op().schema == Schema(
        {"1": dt.int8, "float_col": dt.float64, "length": dt.int32}
    )


def test_select_full_reprojection():
    t1 = t.select(t)
    expected = t.op()
    assert t1.op() == expected


def test_select_across_relations():
    t1 = t.select("bool_col", "int_col", "float_col")
    t2 = t1.select("bool_col", "int_col")
    t3 = t2.select("bool_col")
    expected = Project(parent=t, values={"bool_col": t.bool_col})
    assert t3.op() == expected

    t1 = t.select(t.bool_col, t.int_col, t.float_col)
    t2 = t1.select(t1.bool_col, t1.int_col)
    t3 = t2.select(t2.bool_col)
    expected = Project(parent=t, values={"bool_col": t.bool_col})
    assert t3.op() == expected

    expected = Project(
        t, {"bool_col": t.bool_col, "int_col": t.int_col, "float_col": t.float_col}
    )
    t1 = t.select(t.bool_col, t.int_col, t.float_col)

    t2 = t1.select(t.bool_col, t1.int_col, t1.float_col)
    t2_ = t1.select(t1.bool_col, t1.int_col, t1.float_col)
    assert t2.equals(t2_)

    t3 = t2.select(t.bool_col, t1.int_col, t2.float_col)
    t3_ = t2.select(t2.bool_col, t2.int_col, t2.float_col)
    assert t3.equals(t3_)

    assert t1.op() == expected
    assert t2.op() == expected
    assert t3.op() == expected

    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    expected = Project(
        t,
        {
            "bool_col": ~t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": t.float_col * 3,
        },
    )
    assert t1.op() == expected

    t2 = t1.select(t1.bool_col, t1.int_col, t1.float_col)
    assert t2.op() == expected

    t3 = t2.select(t2.bool_col, t2.int_col, float_col=t2.float_col * 2)
    expected = Project(
        t,
        {
            "bool_col": t.bool_col,
            "int_col": t.int_col + 1,
            "float_col": (t.float_col * 3).name("float_col") * 2,
        },
    )
    assert t3.op() == expected


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


def test_where_after_select():
    filt = t.select(t.bool_col).filter(t.bool_col)
    expected = Filter(
        parent=Project(t, {"bool_col": t.bool_col}), predicates=[t.bool_col]
    )
    assert filt.op() == expected

    filt = t.select(int_col=t.bool_col).filter(t.bool_col)
    expected = Filter(
        parent=Project(t, {"int_col": t.bool_col}), predicates=[t.bool_col]
    )
    assert filt.op() == expected


def test_subsequent_filters_are_squashed():
    filt = t.filter(t.bool_col).filter(t.int_col > 0)
    expected = Filter(parent=t, predicates=[t.bool_col, t.int_col > 0])
    assert filt.op() == expected


def test_subsequent_sorts_are_squashed():
    sort = t.order_by(t.bool_col).order_by(t.int_col)
    expected = Sort(parent=t, keys=[t.bool_col, t.int_col])
    assert sort.op() == expected


def test_projection_before_and_after_filter():
    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    t2 = t1.filter(t1.bool_col)
    t3 = t2.filter(t2.int_col > 0)
    t4 = t3.select(t3.bool_col, t3.int_col)

    assert t4.op() == Project(
        parent=Filter(
            parent=t1,
            predicates=[
                t1.bool_col,
                t1.int_col > 0,
            ],
        ),
        values={
            "bool_col": t1.bool_col,
            "int_col": t1.int_col,
        },
    )


def test_join():
    t1 = UnboundTable("t1", {"a": "int64", "b": "string"}).to_expr()
    t2 = UnboundTable("t2", {"c": "int64", "d": "string"}).to_expr()

    joined = t1.join(t2, [t1.a == t2.c])
    result = joined.finish()

    assert result.op() == Join(
        how="inner",
        left=t1.op(),
        right=t2.op(),
        predicates=[t1.a == t2.c],
        fields={
            "a": t1.a,
            "b": t1.b,
            "c": t2.c,
            "d": t2.d,
        },
    )


def test_chained_join():
    a = UnboundTable("a", {"a": "int64", "b": "string"}).to_expr()
    b = UnboundTable("b", {"c": "int64", "d": "string"}).to_expr()
    c = UnboundTable("c", {"e": "int64", "f": "string"}).to_expr()

    joined = a.join(b, [a.a == b.c]).join(c, [a.a == c.e])
    result = joined.finish()
    assert result.op() == Join(
        how="inner",
        left=Join(
            how="inner",
            left=a.op(),
            right=b.op(),
            predicates=[a.a == b.c],
            fields={
                "a": a.a,
                "b": a.b,
                "c": b.c,
                "d": b.d,
            },
        ),
        right=c.op(),
        predicates=[a.a == c.e],
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
    assert result.op() == Join(
        how="inner",
        left=Join(
            how="inner",
            left=a.op(),
            right=b.op(),
            predicates=[a.a == b.c],
            fields={
                "a": a.a,
                "d": b.d,
                "c": b.c,
            },
        ),
        right=c.op(),
        predicates=[b.c == c.e],
        fields={
            "a": a.a,
            "d": b.d,
            "f": c.f,
        },
    )


def test_chained_join_referencing_intermediate_table():
    a = UnboundTable("a", {"a": "int64", "b": "string"}).to_expr()
    b = UnboundTable("b", {"c": "int64", "d": "string"}).to_expr()
    c = UnboundTable("c", {"e": "int64", "f": "string"}).to_expr()

    ab = a.join(b, [a.a == b.c])
    abc = ab.join(c, [ab.a == c.e])
