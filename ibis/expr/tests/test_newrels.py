from __future__ import annotations

import contextlib

import pytest

import ibis
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.selectors as s
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


@contextlib.contextmanager
def join_tables(table):
    yield [t.to_expr() for t in table.op().tables]


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
    assert isinstance(t, ir.Table)
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
    agg = t.agg([t.a.sum(), t.a.mean()])

    msg = "must have exactly one column, got 2"
    with pytest.raises(IntegrityError, match=msg):
        ops.ScalarSubquery(agg)
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


def test_value_to_array_creates_subquery():
    rel = t.int_col.sum().as_table()
    with pytest.warns(FutureWarning, match="as_scalar"):
        expr = rel.to_array()

    op = expr.op()
    assert op.shape.is_scalar()
    assert op.dtype.is_int64()
    assert isinstance(op, ops.ScalarSubquery)


def test_as_scalar_creates_subquery():
    # scalar literal case
    lit = ibis.literal(1)
    expr = lit.as_scalar()
    assert expr.equals(lit)

    # scalar reduction case
    reduction = t.int_col.sum()
    expr = reduction.as_scalar()
    expected = ops.ScalarSubquery(reduction.as_table())
    assert expr.op() == expected

    # column case
    expr = t.int_col.as_scalar()
    expected = ops.ScalarSubquery(t.int_col.as_table())
    assert expr.op() == expected

    # table case
    proj = t.select(t.int_col)
    expr = proj.as_scalar()
    expected = ops.ScalarSubquery(proj)
    assert expr.op() == expected

    # table case but with multiple columns which can be validated
    with pytest.raises(IntegrityError, match="must have exactly one column"):
        t.as_scalar()


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

    mut = t.mutate(a=ibis.literal(42))
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


def test_order_by_supports_varargs():
    expr = t.order_by(t.int_col, t.float_col)
    expected = ops.Sort(
        parent=t,
        keys=[
            ops.SortKey(ops.Field(t, "int_col"), ascending=True),
            ops.SortKey(ops.Field(t, "float_col"), ascending=True),
        ],
    )
    assert expr.op() == expected

    # same with deferred and string column references
    expr = t.order_by(_.int_col, "float_col")
    assert expr.op() == expected


def test_subsequent_filter():
    f1 = t.filter(t.bool_col)
    f2 = f1.filter(t.int_col > 0)
    expected = Filter(f1, predicates=[f1.int_col > 0])
    assert f2.op() == expected


def test_project_doesnt_dereference_literal_expressions():
    one = ibis.literal(1)
    two = ibis.literal(2)
    four = (one + one) * two
    t1 = t.mutate(four=four)
    assert t1.op() == Project(
        parent=t,
        values={
            "bool_col": t.bool_col,
            "int_col": t.int_col,
            "float_col": t.float_col,
            "string_col": t.string_col,
            "four": four,
        },
    )

    t2 = t1.select(four)
    assert t2.op() == Project(parent=t1, values={four.get_name(): four})


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


def test_join():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})
    joined = t1.join(t2, [t1.a == t2.c])

    assert isinstance(joined, ir.Join)
    assert isinstance(joined.op(), JoinChain)
    assert isinstance(joined.op().to_expr(), ir.Join)

    result = joined._finish()
    assert isinstance(joined, ir.Table)
    assert isinstance(joined.op(), JoinChain)
    assert isinstance(joined.op().to_expr(), ir.Join)

    with join_tables(joined) as (t1, t2):
        assert result.op() == JoinChain(
            first=t1,
            rest=[
                JoinLink("inner", t2, [t1.a == t2.c]),
            ],
            values={
                "a": t1.a,
                "b": t1.b,
                "c": t2.c,
                "d": t2.d,
            },
        )


def test_join_integrity_checks():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})

    # correct example
    r1 = ops.JoinReference(t1, 10)
    r2 = ops.JoinReference(t1, 20)
    assert r1 != r2
    assert hash(r1) != hash(r2)
    chain = ops.JoinChain(r1, [ops.JoinLink("inner", r2, [True])], values={})
    assert isinstance(chain, JoinChain)

    # not unique tables
    r1 = ops.JoinReference(t1, 10)
    r2 = ops.JoinReference(t1, 10)
    assert r1 == r2
    assert hash(r1) == hash(r2)
    with pytest.raises(IntegrityError):
        ops.JoinChain(r1, [ops.JoinLink("inner", r2, [True])], values={})


def test_join_unambiguous_select():
    a = ibis.table(name="a", schema={"a_int": "int64", "a_str": "string"})
    b = ibis.table(name="b", schema={"b_int": "int64", "b_str": "string"})

    join = a.join(b, a.a_int == b.b_int)
    expr1 = join["a_int", "b_int"]
    expr2 = join.select("a_int", "b_int")
    assert expr1.equals(expr2)

    with join_tables(join) as (r1, r2):
        assert expr1.op() == JoinChain(
            first=r1,
            rest=[JoinLink("inner", r2, [r1.a_int == r2.b_int])],
            values={
                "a_int": r1.a_int,
                "b_int": r2.b_int,
            },
        )


def test_join_with_subsequent_projection():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    # a single computed value is pulled to a subsequent projection
    joined = t1.join(t2, [t1.a == t2.c])
    expr = joined.select(t1.a, t1.b, col=t2.c + 1)
    with join_tables(joined) as (r1, r2):
        expected = JoinChain(
            first=r1,
            rest=[JoinLink("inner", r2, [r1.a == r2.c])],
            values={"a": r1.a, "b": r1.b, "col": r2.c + 1},
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
    with join_tables(joined) as (r1, r2):
        expected = JoinChain(
            first=r1,
            rest=[JoinLink("inner", r2, [r1.a == r2.c])],
            values={
                "a": r1.a,
                "b": r1.b,
                "foo": r2.c + 1,
                "bar": r2.c + 2,
                "baz": r2.d.name("bar") + "3",
                "baz2": r2.c + r1.a,
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
    with join_tables(expr) as (r1, r2):
        expected = JoinChain(
            first=r1,
            rest=[JoinLink("inner", r2, [r1.a == r2.a])],
            values={
                "a": r1.a,
                "b": r1.b,
                "foo": r2.a + 1,
                "bar": r1.a + r2.a,
            },
        )
        assert expr.op() == expected


def test_chained_join():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})
    joined = a.join(b, [a.a == b.c]).join(c, [a.a == c.e])
    result = joined._finish()

    with join_tables(joined) as (r1, r2, r3):
        assert result.op() == JoinChain(
            first=r1,
            rest=[
                JoinLink("inner", r2, [r1.a == r2.c]),
                JoinLink("inner", r3, [r1.a == r3.e]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "c": r2.c,
                "d": r2.d,
                "e": r3.e,
                "f": r3.f,
            },
        )

    joined = a.join(b, [a.a == b.c]).join(c, [b.c == c.e])
    result = joined.select(a.a, b.d, c.f)

    with join_tables(joined) as (r1, r2, r3):
        assert result.op() == JoinChain(
            first=r1,
            rest=[
                JoinLink("inner", r2, [r1.a == r2.c]),
                JoinLink("inner", r3, [r2.c == r3.e]),
            ],
            values={
                "a": r1.a,
                "d": r2.d,
                "f": r3.f,
            },
        )


def test_chained_join_referencing_intermediate_table():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})

    ab = a.join(b, [a.a == b.c])
    abc = ab.join(c, [ab.a == c.e])
    result = abc._finish()

    with join_tables(abc) as (r1, r2, r3):
        assert result.op() == JoinChain(
            first=r1,
            rest=[
                JoinLink("inner", r2, [r1.a == r2.c]),
                JoinLink("inner", r3, [r1.a == r3.e]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "c": r2.c,
                "d": r2.d,
                "e": r3.e,
                "f": r3.f,
            },
        )

    assert isinstance(ab, ir.JoinExpr)
    assert isinstance(abc, ir.JoinExpr)


@pytest.mark.parametrize("how", ["positional", "cross"])
def test_chained_join_positional_cross(how):
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})
    joined = a.join(b, how=how).join(c, how=how)
    result = joined._finish()

    with join_tables(joined) as (r1, r2, r3):
        assert result.op() == JoinChain(
            first=r1,
            rest=[JoinLink(how, r2, ()), JoinLink(how, r3, ())],
            values={
                "a": r1.a,
                "b": r1.b,
                "c": r2.c,
                "d": r2.d,
                "e": r3.e,
                "f": r3.f,
            },
        )


def test_join_predicate_dereferencing():
    # See #790, predicate pushdown in joins not supported

    # Star schema with fact table
    table = ibis.table({"c": int, "f": float, "foo_id": str, "bar_id": str})
    table2 = ibis.table({"foo_id": str, "value1": float, "value3": float})
    table3 = ibis.table({"bar_id": str, "value2": float})

    filtered = table.filter(table["f"] > 0)

    # dereference table.foo_id to filtered.foo_id
    j1 = filtered.left_join(table2, table["foo_id"] == table2["foo_id"])
    with join_tables(j1) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("left", r2, [r1.foo_id == r2.foo_id]),
            ],
            values={
                "c": r1.c,
                "f": r1.f,
                "foo_id": r1.foo_id,
                "bar_id": r1.bar_id,
                "foo_id_right": r2.foo_id,
                "value1": r2.value1,
                "value3": r2.value3,
            },
        )
        assert j1.op() == expected

    j1 = filtered.left_join(table2, table["foo_id"] == table2["foo_id"])
    j2 = j1.inner_join(table3, filtered["bar_id"] == table3["bar_id"])
    view = j2.select(filtered, table2["value1"], table3["value2"])
    with join_tables(j2) as (r1, r2, r3):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("left", r2, [r1.foo_id == r2.foo_id]),
                ops.JoinLink("inner", r3, [r1.bar_id == r3.bar_id]),
            ],
            values={
                "c": r1.c,
                "f": r1.f,
                "foo_id": r1.foo_id,
                "bar_id": r1.bar_id,
                "foo_id_right": r2.foo_id,
                "value1": r2.value1,
                "value3": r2.value3,
                "value2": r3.value2,
            },
        )
        assert j2.op() == expected
        assert view.op() == expected.copy(
            values={
                "c": r1.c,
                "f": r1.f,
                "foo_id": r1.foo_id,
                "bar_id": r1.bar_id,
                "value1": r2.value1,
                "value2": r3.value2,
            }
        )


def test_join_predicate_dereferencing_using_tuple_syntax():
    # GH #8292

    t = ibis.table(name="t", schema={"x": "int64", "y": "string"})
    t2 = t.mutate(x=_.x + 1)
    t3 = t.mutate(x=_.x + 1)
    t4 = t.mutate(x=_.x + 2)

    j1 = ibis.join(t2, t3, [(t2.x, t3.x)])
    j2 = ibis.join(t2, t4, [(t2.x, t4.x)])

    with join_tables(j1) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.x == r2.x]),
            ],
            values={
                "x": r1.x,
                "y": r1.y,
                "y_right": r2.y,
            },
        )
        assert j1.op() == expected

    with join_tables(j2) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.x == r2.x]),
            ],
            values={
                "x": r1.x,
                "y": r1.y,
                "y_right": r2.y,
            },
        )
        assert j2.op() == expected


def test_join_with_selector_predicate():
    t1 = ibis.table(name="t1", schema={"a": "string", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "string", "d": "string"})

    joined = t1.join(t2, s.of_type("string"))
    with join_tables(joined) as (r1, r2):
        expected = JoinChain(
            first=r1,
            rest=[
                JoinLink("inner", r2, [r1.a == r2.c, r1.b == r2.d]),
            ],
            values={"a": r1.a, "b": r1.b, "c": r2.c, "d": r2.d},
        )
        assert joined.op() == expected


def test_join_rhs_dereferencing():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})

    t3 = t2.mutate(e=t2.c + 1)
    joined = t1.join(t3, [t1.a == t2.c])
    with join_tables(joined) as (r1, r2):
        expected = JoinChain(
            first=r1,
            rest=[
                JoinLink("inner", r2, [r1.a == r2.c]),
            ],
            values={"a": r1.a, "b": r1.b, "c": r2.c, "d": r2.d, "e": r2.e},
        )
        assert joined.op() == expected

    joined = t1.join(t3, [t1.a == (t2.c + 1)])
    with join_tables(joined) as (r1, r2):
        expected = JoinChain(
            first=r1,
            rest=[
                JoinLink("inner", r2, [r1.a == r2.e]),
            ],
            values={"a": r1.a, "b": r1.b, "c": r2.c, "d": r2.d, "e": r2.e},
        )
        assert joined.op() == expected


def test_aggregate():
    agg = t.aggregate([t.int_col.sum()], by=[t.bool_col])
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
    t4 = t3.join(t3, ["key"])

    with join_tables(t4) as (r1, r2, r3):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.key == r2.key]),
            ],
            values={"key": r1.key},
        )
        assert t3.op() == expected

        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.key == r2.key]),
                ops.JoinLink("inner", r3, [r1.key == r3.key]),
            ],
            values={"key": r1.key},
        )
        assert t4.op() == expected


def test_self_join_view():
    t = ibis.table(schema={"x": "int", "y": "int", "z": "str"})
    t_view = t.view()
    expr = t.join(t_view, t.x == t_view.y).select("x", "y", "z", "z_right")

    with join_tables(expr) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.x == r2.y]),
            ],
            values={"x": r1.x, "y": r1.y, "z": r1.z, "z_right": r2.z},
        )
        assert expr.op() == expected


def test_self_join_with_view_projection():
    t1 = ibis.table(schema={"x": "int", "y": "int", "z": "str"})
    t2 = t1.view()
    expr = t1.inner_join(t2, ["x"]).select(t1)

    with join_tables(expr) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.x == r2.x]),
            ],
            values={"x": r1.x, "y": r1.y, "z": r1.z},
        )
        assert expr.op() == expected


def test_joining_same_table_twice():
    left = ibis.table(name="left", schema={"time1": int, "value": float, "a": str})
    right = ibis.table(name="right", schema={"time2": int, "value2": float, "b": str})

    first = left.inner_join(right, left.a == right.b)

    with pytest.raises(IbisInputError, match="Ambiguous field reference"):
        first.inner_join(right, left.value == right.value2)

    right_ = right.view()
    second = first.inner_join(right_, left.value == right_.value2)
    with join_tables(second) as (r1, r2, r3):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.b]),
                ops.JoinLink("inner", r3, [r1.value == r3.value2]),
            ],
            values={
                "time1": r1.time1,
                "value": r1.value,
                "a": r1.a,
                "time2": r2.time2,
                "value2": r2.value2,
                "b": r2.b,
                "time2_right": r3.time2,
                "value2_right": r3.value2,
                "b_right": r3.b,
            },
        )
        assert second.op() == expected


def test_join_chain_gets_reused_and_continued_after_a_select():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    b = ibis.table(name="b", schema={"c": "int64", "d": "string"})
    c = ibis.table(name="c", schema={"e": "int64", "f": "string"})

    ab = a.join(b, [a.a == b.c])
    abc = ab.select(a.b, b.d).join(c, [a.a == c.e])

    with join_tables(abc) as (r1, r2, r3):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.c]),
                ops.JoinLink("inner", r3, [r1.a == r3.e]),
            ],
            values={
                "b": r1.b,
                "d": r2.d,
                "e": r3.e,
                "f": r3.f,
            },
        )
        assert abc.op() == expected


def test_self_join_extensive():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})

    with pytest.raises(IbisInputError, match="Ambiguous field reference"):
        a.join(a, [a.a == a.a])

    a_ = a.view()
    aa = a.join(a_, [a.a == a_.a])
    with join_tables(aa) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "b_right": r2.b,
            },
        )
        assert aa.op() == expected

    aa1 = a.join(a, "a")
    aa2 = a.join(a, [("a", "a")])
    with join_tables(aa1) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "b_right": r2.b,
            },
        )
        assert aa1.op() == expected
        assert aa2.op() == expected

    aaa = a.join(a, "a").join(a, "a")
    aaa1 = aa1.join(a, "a")
    aaa2 = aa1.join(a, [("a", "a")])
    with join_tables(aaa) as (r1, r2, r3):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
                ops.JoinLink("inner", r3, [r1.a == r3.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "b_right": r2.b,
            },
        )
        assert aaa.op() == expected
        assert aaa1.op() == expected
        assert aaa2.op() == expected


def test_self_join_with_intermediate_selection():
    a = ibis.table(name="a", schema={"a": "int64", "b": "string"})
    proj = a[["b", "a"]]
    # the predicate only references the original table, unless we enforce
    # that the predicates must contain both sides of the join, we can't
    # do much with this, perhaps raise a warning
    join = proj.join(a, [a.a == a.a])
    with join_tables(join) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r2.a == r2.a]),
            ],
            values={
                "b": r1.b,
                "a": r1.a,
                "a_right": r2.a,
                "b_right": r2.b,
            },
        )
        assert join.op() == expected

    join = proj.join(a, [proj.a == a.a])
    with join_tables(join) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
            ],
            values={
                "b": r1.b,
                "a": r1.a,
                "b_right": r2.b,
            },
        )
        assert join.op() == expected

    a1 = a.view()
    a2 = a.view()
    aa = a.join(a1, [a.a == a1.a])["a", "b_right"]
    aaa = aa.join(a2, [aa.a == a2.a])
    with join_tables(aaa) as (r1, r2, r3):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
                ops.JoinLink("inner", r3, [r1.a == r3.a]),
            ],
            values={
                "a": r1.a,
                "b_right": r2.b,
                "b": r3.b,
            },
        )
        assert aaa.op() == expected

    # TODO(kszucs): this use case could be supported if `_get_column` gets
    # overridden to return underlying column reference, but that would mean
    # that `aa.a` returns with `a.a` instead of `aa.a` which breaks other
    # things; the other possible solution is to use 2way dereferencing
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


def test_self_view_join_followed_by_aggregate_correctly_dereference_fields():
    t = ibis.table(
        name="t", schema={"a": "int64", "b": "int64", "f": "int64", "g": "string"}
    )

    agged = t.aggregate([t.f.sum().name("total")], by=["g", "a", "b"])
    view = agged.view()
    metrics = [(agged.total - view.total).max().name("metric")]
    join = agged.inner_join(view, [agged.a == view.b])
    agg = join.aggregate(metrics, by=[agged.g])

    with join_tables(join) as (r1, r2):
        expected_join = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.b]),
            ],
            values={
                "g": r1.g,
                "a": r1.a,
                "b": r1.b,
                "total": r1.total,
                "g_right": r2.g,
                "a_right": r2.a,
                "b_right": r2.b,
                "total_right": r2.total,
            },
        ).to_expr()
        expected_agg = ops.Aggregate(
            parent=join,
            groups={
                "g": join.g,
            },
            metrics={
                "metric": (join.total - join.total_right).max(),
            },
        ).to_expr()
        assert join.equals(expected_join)
        assert agg.equals(expected_agg)


def test_join_expressions_are_equal():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "int64"})
    t2 = ibis.table(name="t2", schema={"a": "int64", "b": "int64"})

    join1 = t1.inner_join(t2, [t1.a == t2.a])
    join2 = t1.inner_join(t2, [t1.a == t2.a])
    assert join1.equals(join2)


def test_join_between_joins():
    t1 = ibis.table(
        [("key1", "string"), ("key2", "string"), ("value1", "double")],
        "first",
    )
    t2 = ibis.table([("key1", "string"), ("value2", "double")], "second")
    t3 = ibis.table(
        [("key2", "string"), ("key3", "string"), ("value3", "double")],
        "third",
    )
    t4 = ibis.table([("key3", "string"), ("value4", "double")], "fourth")

    left = t1.inner_join(t2, [("key1", "key1")]).select(t1, t2.value2)
    right = t3.inner_join(t4, [("key3", "key3")]).select(t3, t4.value4)

    joined = left.inner_join(right, left.key2 == right.key2)

    # At one point, the expression simplification was resulting in bad refs
    # here (right.value3 referencing the table inside the right join)
    exprs = [left, right.value3, right.value4]
    expr = joined.select(exprs)

    with join_tables(expr) as (r1, r2, r3):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.key1 == r2.key1]),
                ops.JoinLink("inner", r3, [r1.key2 == r3.key2]),
            ],
            values={
                "key1": r1.key1,
                "key2": r1.key2,
                "value1": r1.value1,
                "value2": r2.value2,
                "value3": r3.value3,
                "value4": r3.value4,
            },
        )
        assert expr.op() == expected


def test_join_with_filtered_join_of_left():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"a": "int64", "b": "string"})

    joined = t1.left_join(t2, [t1.a == t2.a]).filter(t1.a < 5)
    expr = t1.left_join(joined, [t1.a == joined.a]).select(t1)

    with join_tables(expr) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("left", r2, [r1.a == r2.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
            },
        )
        assert expr.op() == expected


def test_join_method_docstrings():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"c": "int64", "d": "string"})
    joined = t1.join(t2, [t1.a == t2.c])

    assert isinstance(t1, ir.Table)
    assert isinstance(joined, ir.Join)
    assert isinstance(joined, ir.Table)

    method_names = [
        "select",
        "join",
        "inner_join",
        "left_join",
        "outer_join",
        "semi_join",
        "anti_join",
        "asof_join",
        "cross_join",
        "right_join",
        "any_inner_join",
        "any_left_join",
    ]
    for method in method_names:
        join_method = getattr(joined, method)
        table_method = getattr(t1, method)
        assert join_method.__doc__ == table_method.__doc__


def test_join_with_compound_predicate():
    t1 = ibis.table(name="t", schema={"a": "string", "b": "string"})
    t2 = t1.view()

    joined = t1.join(
        t2,
        [
            t1.a == t2.a,
            (t1.a != t2.b) | (t1.b != t2.a),
            (t1.a != t2.b) ^ (t1.b != t2.a),
            (t1.a != t2.b) & (t1.b != t2.a),
            (t1.a + t1.a != t2.b) & (t1.b + t1.b != t2.a),
        ],
    )
    expr = joined.select(t1)
    with join_tables(joined) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink(
                    "inner",
                    r2,
                    [
                        r1.a == r2.a,
                        (r1.a != r2.b) | (r1.b != r2.a),
                        (r1.a != r2.b) ^ (r1.b != r2.a),
                        # these are flattened
                        r1.a != r2.b,
                        r1.b != r2.a,
                        r1.a + r1.a != r2.b,
                        r1.b + r1.b != r2.a,
                    ],
                ),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
            },
        )
        assert expr.op() == expected


def test_inner_join_convenience():
    t1 = ibis.table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = ibis.table(name="t2", schema={"a": "int64", "c": "string"})
    t3 = ibis.table(name="t3", schema={"a": "int64", "d": "string"})
    t4 = ibis.table(name="t4", schema={"a": "int64", "e": "string"})
    t5 = ibis.table(name="t5", schema={"a": "int64", "f": "string"})

    first_join = t1.inner_join(t2, [t1.a == t2.a])
    with join_tables(first_join) as (r1, r2):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "c": r2.c,
            },
        )
        # finish to evaluate the collisions
        result = first_join._finish().op()
        assert result == expected

    # note that we are joining on r2.a which isn't among the values
    second_join = first_join.inner_join(t3, [r2.a == t3.a])
    with join_tables(second_join) as (r1, r2, r3):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
                ops.JoinLink("inner", r3, [r2.a == r3.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "c": r2.c,
                "d": r3.d,
            },
        )
        # finish to evaluate the collisions
        result = second_join._finish().op()
        assert result == expected

    third_join = second_join.left_join(t4, [r3.a == t4.a])
    with join_tables(third_join) as (r1, r2, r3, r4):
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
                ops.JoinLink("inner", r3, [r2.a == r3.a]),
                ops.JoinLink("left", r4, [r3.a == r4.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "c": r2.c,
                "d": r3.d,
                "a_right": r4.a,
                "e": r4.e,
            },
        )
        # finish to evaluate the collisions
        result = third_join._finish().op()
        assert result == expected

    fourth_join = third_join.inner_join(t5, [r3.a == t5.a], rname="{name}_")
    with join_tables(fourth_join) as (r1, r2, r3, r4, r5):
        # equality groups are being reset
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
                ops.JoinLink("inner", r3, [r2.a == r3.a]),
                ops.JoinLink("left", r4, [r3.a == r4.a]),
                ops.JoinLink("inner", r5, [r3.a == r5.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "c": r2.c,
                "d": r3.d,
                "a_right": r4.a,
                "e": r4.e,
                "f": r5.f,
            },
        )
        # finish to evaluate the collisions
        result = fourth_join._finish().op()
        assert result == expected

    with pytest.raises(IntegrityError):
        # equality groups are being reset, t5.a would be renamed to 'a_right'
        # which already exists
        third_join.inner_join(t5, [r4.a == t5.a])._finish()

    fifth_join = third_join.inner_join(t5, [r4.a == t5.a], rname="{name}_")
    with join_tables(fifth_join) as (r1, r2, r3, r4, r5):
        # equality groups are being reset
        expected = ops.JoinChain(
            first=r1,
            rest=[
                ops.JoinLink("inner", r2, [r1.a == r2.a]),
                ops.JoinLink("inner", r3, [r2.a == r3.a]),
                ops.JoinLink("left", r4, [r3.a == r4.a]),
                ops.JoinLink("inner", r5, [r4.a == r5.a]),
            ],
            values={
                "a": r1.a,
                "b": r1.b,
                "c": r2.c,
                "d": r3.d,
                "a_right": r4.a,
                "e": r4.e,
                "a_": r5.a,
                "f": r5.f,
            },
        )
        # finish to evaluate the collisions
        result = fifth_join._finish().op()
        assert result == expected


def test_subsequent_order_by_calls():
    ts = t.order_by(ibis.desc("int_col")).order_by("int_col")
    first = ops.Sort(t, [t.int_col.desc()]).to_expr()
    second = ops.Sort(first, [first.int_col.asc()]).to_expr()
    assert ts.equals(second)


@pytest.mark.parametrize("func", [ibis.random, ibis.uuid])
def test_impure_operation_dereferencing(func):
    t = ibis.table({"x": "int64"}, name="t")

    impure = func()
    t1 = t.mutate(y=impure)
    t2 = t1.mutate(z=impure.cast("string"))

    expected = ops.Project(
        parent=t1,
        values={"x": t1.x, "y": t1.y, "z": impure.cast("string")},
    )
    assert t2.op() == expected

    v1 = func()
    v2 = func()

    t1 = t.mutate(y=v1)
    t2 = t1.mutate(z=v2.cast("string"))
    expected = ops.Project(
        parent=t1, values={"x": t1.x, "y": t1.y, "z": v2.cast("string")}
    )
    assert t2.op() == expected


def test_mutate_ambiguty_check_not_too_strict():
    t = ibis.table({"id": "int64"}, name="t")

    first = t.mutate(v=t["id"])
    second = first.mutate(v2=t["id"])
    expected = ops.Project(
        parent=first,
        values={"id": first.id, "v": first.v, "v2": first.id},
    )
    assert second.op() == expected


def test_projections_with_different_field_order_are_unequal():
    t = ibis.table({"a": "int64", "b": "string"}, name="t")

    t1 = t.select(a=1, b=2)
    t2 = t.select(b=2, a=1)

    assert not t1.equals(t2)


def test_filters_are_allowed_to_have_the_same_name():
    t = ibis.table({"a": "int64", "b": "string"}, name="t")
    f1 = t.filter(t.a > 1, t.a > 1)
    f2 = t.filter(t.a > 1)
    f3 = t.filter((t.a > 1).name("a"))
    f4 = t.filter((t.a > 1).name("a"), (t.a > 1).name("b"))
    assert f1.equals(f2)
    assert f1.equals(f3)
    assert f1.equals(f4)


def test_projections_with_similar_expressions_have_different_names():
    t = ibis.table({"a": "string", "b": "string"}, name="t")

    a = t.a.fill_null("")
    b = t.b.fill_null("")
    assert a.op().name != b.op().name

    expr = t.select(a, b)
    fields = expr.op().fields

    assert a.op().name in fields
    assert b.op().name in fields

    assert expr.schema() == ibis.schema({a.op().name: "string", b.op().name: "string"})


def test_expr_in_join_projection():
    t1 = ibis.table({"a": "int64", "b": "string"}, name="t1")
    t2 = ibis.table({"c": "int64", "b": "string"}, name="t2")
    t3 = ibis.table({"a": "int64", "d": "int64", "e": "string"}, name="t3")
    expr = t1.inner_join(t2, "b").select(
        "a", lit1=1, lit2=2 * t1.a, lit3=_.c - 5, lit4=t2.b.length() / 2.0
    )

    op = expr.op()
    assert isinstance(op, ops.JoinChain)
    assert op.schema == ibis.schema(
        {
            "a": "int64",
            "lit1": "int8",
            "lit2": "int64",
            "lit3": "int64",
            "lit4": "float64",
        }
    )

    # simple chain selection
    expr2 = expr.inner_join(t3, "a").select("a", "d", "lit1", "lit2", "lit3", "lit4")
    op2 = expr2.op()
    assert isinstance(op2, ops.JoinChain)
    assert op2.schema == ibis.schema(
        {
            "a": "int64",
            "d": "int64",
            "lit1": "int8",
            "lit2": "int64",
            "lit3": "int64",
            "lit4": "float64",
        }
    )

    # chain with expressions from all three tables in the join
    expr = (
        t1.inner_join(t2, "b")
        .inner_join(t3, t2.b == t3.e)
        .select(
            "a",
            lit1=1,
            lit2=2 * t1.a,
            lit3=_.c - 5,
            lit4=t2.b.length() / 2.0,
            lit5=_.e.cast("int") * 3,
            lit6=t3.d + 1,
        )
    )

    op = expr.op()
    assert isinstance(op, ops.JoinChain)
    assert op.schema == ibis.schema(
        {
            "a": "int64",
            "lit1": "int8",
            "lit2": "int64",
            "lit3": "int64",
            "lit4": "float64",
            "lit5": "int64",
            "lit6": "int64",
        }
    )


def test_analytic_dereference():
    t = ibis.table({"a": "int"})
    ix = ibis.row_number()
    expr = t.mutate(ix=ix).filter(ix == 5)
    assert expr.op().predicates == (
        ops.Equals(ops.WindowFunction(ops.RowNumber()), ops.Literal(5, dtype="int8")),
    )
