from __future__ import annotations

import pytest

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from ibis.common.exceptions import IntegrityError
from ibis.expr.newrels import Field, Filter, Project, Sort, TableExpr, UnboundTable
from ibis.expr.schema import Schema

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
    with pytest.raises(IntegrityError):
        t1.select(t.bool_col, t1.int_col, t1.float_col)
    t2 = t1.select(t1.bool_col, t1.int_col, t1.float_col)
    with pytest.raises(IntegrityError):
        t2.select(t.bool_col, t1.int_col, t2.float_col)
    t3 = t2.select(t2.bool_col, t2.int_col, t2.float_col)
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
    filt = t.where(t.bool_col)
    expected = Filter(parent=t, predicates=[t.bool_col])
    assert filt.op() == expected

    filt = t.where(t.bool_col, t.int_col > 0)
    expected = Filter(parent=t, predicates=[t.bool_col, t.int_col > 0])
    assert filt.op() == expected


def test_subsequent_filters_are_squashed():
    filt = t.where(t.bool_col).where(t.int_col > 0)
    expected = Filter(parent=t, predicates=[t.bool_col, t.int_col > 0])
    assert filt.op() == expected


def test_subsequent_sorts_are_squashed():
    sort = t.order_by(t.bool_col).order_by(t.int_col)
    expected = Sort(parent=t, keys=[t.bool_col, t.int_col])
    assert sort.op() == expected


def test_e():
    t1 = t.select(
        bool_col=~t.bool_col, int_col=t.int_col + 1, float_col=t.float_col * 3
    )
    t2 = t1.where(t1.bool_col)
    t3 = t2.where(t2.int_col > 0)
    t4 = t3.select(t3.bool_col, t3.int_col)
