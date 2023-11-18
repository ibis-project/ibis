from __future__ import annotations

import pytest

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import _
from ibis.common.exceptions import IntegrityError
from ibis.expr.operations import (
    Aggregate,
    Field,
    Filter,
    ForeignField,
    Join,
    JoinLink,
    Project,
    UnboundTable,
    table,
)
from ibis.expr.schema import Schema
import ibis.expr.types as ir

# TODO(kszucs):
# def test_relation_coercion()
# def test_where_flattens_predicates()

t = table(
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


def test_where_after_select():
    t1 = t.select(t.bool_col)
    t2 = t1.filter(t.bool_col)
    expected = Filter(parent=t1, predicates=[t1.bool_col])
    assert t2.op() == expected

    t1 = t.select(int_col=t.bool_col)
    t2 = t1.filter(t.bool_col)
    expected = Filter(parent=t1, predicates=[t1.int_col])
    assert t2.op() == expected


def test_subsequent_filter():
    f1 = t.filter(t.bool_col)
    f2 = f1.filter(t.int_col > 0)
    expected = Filter(f1, predicates=[f1.int_col > 0])
    assert f2.op() == expected

    f2_opt = f2.optimize()
    assert f2_opt.op() == Filter(t, predicates=[t.bool_col, t.int_col > 0])


def test_projection_before_and_after_filter():
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

    t2_opt = t2.optimize()
    assert t2_opt.op() == Filter(parent=t1, predicates=[t1.bool_col])

    t3_opt = t3.optimize()
    assert t3_opt.op() == Filter(parent=t1, predicates=[t1.bool_col, t1.int_col > 0])

    t4_opt = t4.optimize()
    assert t4_opt.op() == Project(
        parent=t3_opt, values={"bool_col": t3_opt.bool_col, "int_col": t3_opt.int_col}
    )


def test_foreign_field_identification():
    t1 = t.filter(t.bool_col)
    t2 = t.select(summary=t1.int_col.sum())
    node = t2.op().fields["summary"].arg
    assert isinstance(node, ForeignField)


# TODO(kszucs): add test for failing integrity checks
def test_join():
    t1 = table("t1", {"a": "int64", "b": "string"})
    t2 = table("t2", {"c": "int64", "d": "string"})

    joined = t1.join(t2, [t1.a == t2.c])
    result = joined.finish()
    assert result.op() == Join(
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


def test_chained_join():
    a = table("a", {"a": "int64", "b": "string"})
    b = table("b", {"c": "int64", "d": "string"})
    c = table("c", {"e": "int64", "f": "string"})

    joined = a.join(b, [a.a == b.c]).join(c, [a.a == c.e])
    result = joined.finish()
    assert result.op() == Join(
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
    assert result.op() == Join(
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
    a = table("a", {"a": "int64", "b": "string"})
    b = table("b", {"c": "int64", "d": "string"})
    c = table("c", {"e": "int64", "f": "string"})

    ab = a.join(b, [a.a == b.c])
    assert isinstance(ab, ir.JoinExpr)

    assert ab.a.op() == Field(a, "a")
    abc = ab.join(c, [ab.a == c.e])
    assert isinstance(abc, ir.JoinExpr)

    result = abc.finish()
    assert result.op() == Join(
        first=a,
        rest=[JoinLink("inner", b, [a.a == b.c]), JoinLink("inner", c, [a.a == c.e])],
        fields={"a": a.a, "b": a.b, "c": b.c, "d": b.d, "e": c.e, "f": c.f},
    )


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


def test_select_with_uncorrelated_scalar_subquery():
    t1 = table("t1", {"a": "int64", "b": "string"})
    t2 = table("t2", {"c": "int64", "d": "string"})

    # Create a subquery
    t2_filt = t2.filter(t2.d == "value")

    # Ensure that the integrity checks require scalar shaped foreign values
    with pytest.raises(IntegrityError):
        t1.select(t2_filt.c)

    # Use the subquery in an IN condition
    sub = t1.select(t1.a, summary=t2_filt.c.sum())

    assert sub.op() == Project(
        parent=t1,
        values={"a": t1.a, "summary": ops.Sum(ForeignField(rel=t2_filt, name="c"))},
    )


def test_select_with_correlated_scalar_subquery():
    # Define your tables
    t1 = table(name="t1", schema={"a": "int64", "b": "string"})
    t2 = table(name="t2", schema={"c": "int64", "d": "string"})

    # Create a subquery
    filt = t2.filter(t2.d == t1.b)
    summary = filt.c.sum().name("summary")

    # Use the subquery in a select operation
    expr = t1.select(t1.a, summary)
    assert expr.op() == Project(
        parent=t1,
        values={
            "a": t1.a,
            "summary": ops.Sum(
                ForeignField(
                    rel=filt,
                    name="c",
                )
            ),
        },
    )


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
