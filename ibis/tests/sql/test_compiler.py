from __future__ import annotations

import datetime

import ibis
from ibis.backends.base.sql.compiler import Compiler
from ibis.tests.sql.conftest import to_sql
from ibis.tests.util import assert_decompile_roundtrip


def test_union(union, snapshot):
    snapshot.assert_match(to_sql(union), "out.sql")
    assert_decompile_roundtrip(union, snapshot, check_equality=False)


def test_union_project_column(union_all, snapshot):
    # select a column, get a subquery
    expr = union_all[[union_all.key]]
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)


def test_table_intersect(intersect, snapshot):
    snapshot.assert_match(to_sql(intersect), "out.sql")
    assert_decompile_roundtrip(intersect, snapshot, check_equality=False)


def test_table_difference(difference, snapshot):
    snapshot.assert_match(to_sql(difference), "out.sql")
    assert_decompile_roundtrip(difference, snapshot, check_equality=False)


def test_intersect_project_column(intersect, snapshot):
    # select a column, get a subquery
    expr = intersect[[intersect.key]]
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)


def test_difference_project_column(difference, snapshot):
    # select a column, get a subquery
    expr = difference[[difference.key]]
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)


def test_table_distinct(con, snapshot):
    t = con.table("functional_alltypes")

    expr = t[t.string_col, t.int_col].distinct()
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot)


def test_column_distinct(con, snapshot):
    t = con.table("functional_alltypes")
    expr = t[t.string_col].distinct()
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot)


def test_count_distinct(con, snapshot):
    t = con.table("functional_alltypes")

    metric = t.int_col.nunique().name("nunique")
    expr = t[t.bigint_col > 0].group_by("string_col").aggregate([metric])
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot)


def test_multiple_count_distinct(con, snapshot):
    # Impala and some other databases will not execute multiple
    # count-distincts in a single aggregation query. This error reporting
    # will be left to the database itself, for now.
    t = con.table("functional_alltypes")
    metrics = [
        t.int_col.nunique().name("int_card"),
        t.smallint_col.nunique().name("smallint_card"),
    ]

    expr = t.group_by("string_col").aggregate(metrics)
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot)


def test_pushdown_with_or(snapshot):
    t = ibis.table(
        [
            ("double_col", "float64"),
            ("string_col", "string"),
            ("int_col", "int32"),
            ("float_col", "float32"),
        ],
        "functional_alltypes",
    )
    subset = t[(t.double_col > 3.14) & t.string_col.contains("foo")]
    expr = subset[(subset.int_col - 1 == 0) | (subset.float_col <= 1.34)]
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_having_size(snapshot):
    t = ibis.table(
        [
            ("double_col", "double"),
            ("string_col", "string"),
            ("int_col", "int32"),
            ("float_col", "float"),
        ],
        "functional_alltypes",
    )
    expr = t.group_by(t.string_col).having(t.double_col.max() == 1).size()
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_having_from_filter(snapshot):
    t = ibis.table([("a", "int64"), ("b", "string")], "t")
    filt = t[t.b == "m"]
    gb = filt.group_by(filt.b)
    having = gb.having(filt.a.max() == 2)
    expr = having.aggregate(filt.a.sum().name("sum"))
    snapshot.assert_match(to_sql(expr), "out.sql")
    # params get different auto incremented counter identifiers
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)


def test_simple_agg_filter(snapshot):
    t = ibis.table([("a", "int64"), ("b", "string")], name="my_table")
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_agg_and_non_agg_filter(snapshot):
    t = ibis.table([("a", "int64"), ("b", "string")], name="my_table")
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    expr = expr[expr.b == "a"]
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_agg_filter(snapshot):
    t = ibis.table([("a", "int64"), ("b", "int64")], name="my_table")
    t = t.mutate(b2=t.b * 2)
    t = t[["a", "b2"]]
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max().name("blah")]
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_agg_filter_with_alias(snapshot):
    t = ibis.table([("a", "int64"), ("b", "int64")], name="my_table")
    t = t.mutate(b2=t.b * 2)
    t = t[["a", "b2"]]
    filt = t[t.a < 100]
    expr = filt[filt.a.name("A") == filt.a.max().name("blah")]
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_table_drop_with_filter(snapshot):
    left = ibis.table(
        [("a", "int64"), ("b", "string"), ("c", "timestamp")], name="t"
    ).rename(C="c")
    left = left.filter(left.C == datetime.datetime(2018, 1, 1))
    left = left.drop("C")
    left = left.mutate(the_date=datetime.datetime(2018, 1, 1))

    right = ibis.table([("b", "string")], name="s")
    joined = left.join(right, left.b == right.b)
    joined = joined[left.a]
    expr = joined.filter(joined.a < 1.0)
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)


def test_table_drop_consistency():
    # GH2829
    t = ibis.table([("a", "int64"), ("b", "string"), ("c", "timestamp")], name="t")

    expected = t.select(["a", "c"])
    result = t.drop("b")

    assert expected.schema() == result.schema()
    assert set(result.columns) == {"a", "c"}


def test_subquery_where_location(snapshot):
    t = ibis.table(
        [
            ("float_col", "float32"),
            ("timestamp_col", "timestamp"),
            ("int_col", "int32"),
            ("string_col", "string"),
        ],
        name="alltypes",
    )
    param = ibis.param("timestamp").name("my_param")
    expr = (
        t[["float_col", "timestamp_col", "int_col", "string_col"]][
            lambda t: t.timestamp_col < param
        ]
        .group_by("string_col")
        .aggregate(foo=lambda t: t.float_col.sum())
        .foo.count()
    )
    out = Compiler.to_sql(expr, params={param: "20140101"})
    snapshot.assert_match(out, "out.sql")
    # params get different auto incremented counter identifiers
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)


def test_column_expr_retains_name(snapshot):
    t = ibis.table([("int_col", "int32")], name="int_col_table")
    expr = (t.int_col + 4).name("foo")
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot)


def test_column_expr_default_name(snapshot):
    t = ibis.table([("int_col", "int32")], name="int_col_table")
    expr = t.int_col + 4
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot)


def test_union_order_by(snapshot):
    t = ibis.table(dict(a="int", b="string"), name="t")
    expr = t.order_by("b").union(t.order_by("b"))
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)
