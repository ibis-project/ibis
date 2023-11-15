from __future__ import annotations

import pytest

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from ibis import _
from ibis.common.exceptions import IntegrityError
from ibis.config import options
from ibis.expr.newrels import (
    Aggregate,
    Field,
    Filter,
    Join,
    JoinProject,
    Project,
    TableExpr,
    UnboundTable,
)
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

    proj = t.select(_.int_col, myint=_.int_col)
    expected = Project(parent=t, values={"int_col": t.int_col, "myint": t.int_col})
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


def test_select_full_reprojection():
    with options(eager_optimization=False):
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

    with options(eager_optimization=True):
        t1 = t.select(t)
        assert t1.op() == t.op()


def test_subsequent_selections_with_field_names():
    with options(eager_optimization=False):
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

    with options(eager_optimization=True):
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
            parent=t,
            values={
                "bool_col": t.bool_col,
                "int_col": t.int_col,
            },
        )
        t3 = t2.select("bool_col")
        assert t3.op() == Project(
            parent=t,
            values={
                "bool_col": t.bool_col,
            },
        )


def test_subsequent_selections_field_dereferencing():
    with options(eager_optimization=False):
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
        t2_ = t1.select(t.bool_col, t.int_col)
        expected = Project(
            parent=t1,
            values={
                "bool_col": t1.bool_col,
                "int_col": t1.int_col,
            },
        )
        assert t2.op() == expected
        assert t2_.op() == expected

        t3 = t2.select(t2.bool_col)
        t3_ = t2.select(t1.bool_col)
        t3__ = t2.select(t.bool_col)
        expected = Project(
            parent=t2,
            values={
                "bool_col": t2.bool_col,
            },
        )
        assert t3.op() == expected
        assert t3_.op() == expected
        assert t3__.op() == expected

    with options(eager_optimization=True):
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
        t2_ = t1.select(t.bool_col, t.int_col)
        expected = Project(
            parent=t,
            values={
                "bool_col": t.bool_col,
                "int_col": t.int_col,
            },
        )
        assert t2.op() == expected
        assert t2_.op() == expected

        t3 = t2.select(t2.bool_col)
        t3_ = t2.select(t1.bool_col)
        t3__ = t2.select(t.bool_col)
        expected = Project(
            parent=t,
            values={
                "bool_col": t.bool_col,
            },
        )
        assert t3.op() == expected
        assert t3_.op() == expected
        assert t3__.op() == expected

    with options(eager_optimization=True):
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

        t3 = t2.select(
            t2.bool_col,
            t2.int_col,
            float_col=t2.float_col * 2,
            another_col=t1.float_col - 1,
        )
        expected = Project(
            t,
            {
                "bool_col": ~t.bool_col,
                "int_col": t.int_col + 1,
                "float_col": (t.float_col * 3) * 2,
                "another_col": (t.float_col * 3) - 1,
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
    t1 = t.select(t.bool_col)
    t2 = t1.filter(t.bool_col)
    expected = Filter(parent=t1, predicates=[t1.bool_col])
    assert t2.op() == expected

    t1 = t.select(int_col=t.bool_col)
    t2 = t1.filter(t.bool_col)
    expected = Filter(parent=t1, predicates=[t1.int_col])
    assert t2.op() == expected


def test_subsequent_filters_are_squashed():
    filt = t.filter(t.bool_col).filter(t.int_col > 0)
    expected = Filter(parent=t, predicates=[t.bool_col, t.int_col > 0])
    assert filt.op() == expected


# def test_subsequent_sorts_are_squashed():
#     sort = t.order_by(t.bool_col).order_by(t.int_col)
#     expected = Sort(parent=t, keys=[t.bool_col, t.int_col])
#     assert sort.op() == expected


def test_projection_before_and_after_filter():
    with options(eager_optimization=False):
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
        t3_ = t2.filter(t1.int_col > 0)
        assert t3.op() == Filter(parent=t2, predicates=[t2.int_col > 0])
        assert t3_.op() == Filter(parent=t2, predicates=[t2.int_col > 0])

        t4 = t3.select(t3.bool_col, t3.int_col)
        assert t4.op() == Project(
            parent=t3,
            values={
                "bool_col": t3.bool_col,
                "int_col": t3.int_col,
            },
        )

    with options(eager_optimization=True):
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
        assert t3.op() == Filter(parent=t1, predicates=[t1.bool_col, t1.int_col > 0])

        t4 = t3.select(t3.bool_col, t3.int_col)
        assert t4.op() == Project(
            parent=t3, values={"bool_col": t3.bool_col, "int_col": t3.int_col}
        )


# TODO(kszucs): add test for integrity checks
def test_join():
    t1 = UnboundTable("t1", {"a": "int64", "b": "string"}).to_expr()
    t2 = UnboundTable("t2", {"c": "int64", "d": "string"}).to_expr()

    joined = t1.join(t2, [t1.a == t2.c])
    result = joined.finish()
    assert result.op() == JoinProject(
        first=t1,
        rest=[
            Join("inner", t2, [t1.a == t2.c]),
        ],
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
    assert result.op() == JoinProject(
        first=a,
        rest=[
            Join("inner", b, [a.a == b.c]),
            Join("inner", c, [a.a == c.e]),
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
    assert result.op() == JoinProject(
        first=a,
        rest=[
            Join("inner", b, [a.a == b.c]),
            Join("inner", c, [b.c == c.e]),
        ],
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
    with pytest.raises(IntegrityError):
        ab.join(c, [ab.a == c.e])


def test_aggregate():
    agg = t.aggregate(groups=[t.bool_col], metrics=[t.int_col.sum()])
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


# def test_isin_subquery():
#     import ibis

#     # Define your tables
#     t1 = UnboundTable("t1", {"a": "int64", "b": "string"}).to_expr()
#     t2 = UnboundTable("t2", {"c": "int64", "d": "string"}).to_expr()

#     # Create a subquery
#     t2_filt = t2.filter(t2.d == "value")

#     # Use the subquery in an IN condition
#     expr = t1.filter(t1.a.isin(t2_filt.c))

#     print(expr)


# def test_select_with_uncorrelated_scalar_subquery():
#     import ibis

#     # Define your tables
#     t1 = UnboundTable("t1", {"a": "int64", "b": "string"}).to_expr()
#     t2 = UnboundTable("t2", {"c": "int64", "d": "string"}).to_expr()

#     # Create a subquery
#     t2_filt = t2.filter(t2.d == "value")

#     print(t2_filt.c)
#     return

#     # Use the subquery in an IN condition
#     expr = t1.select(t1.a, t2_filt.c.sum())

#     print(expr)


# t1 = ibis.table(name="a", schema={"a": "int64", "b": "string"})
# t2 = ibis.table(name="b", schema={"c": "int64", "d": "string"})
# t3 = ibis.table(name="c", schema={"e": "int64", "f": "string"})

# t1.select(t1.a, t2.c.sum())  # OK
# t1.select(t1.a, (t2.c == t3.e).sum())  # ???


# SELECT name, salary
# FROM employees
# WHERE salary > (SELECT AVG(salary) FROM employees);


# Filter(
#     parent=Project(
#         parent=employees, values={"name": employees.name, "salary": employees.salary}
#     ),
#     predicates=[
#         Greater(
#             employees.salary,
#             Field(
#                 Aggregate(
#                     parent=employees,
#                     groups={},
#                     metrics={"AVG(salary)": employees.salary.mean()},
#                 ),
#                 name="AVG(salary)",
#             )
#         )
#     ],
# )
