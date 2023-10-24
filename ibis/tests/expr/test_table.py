from __future__ import annotations

import datetime
import pickle
import re

import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.selectors as s
from ibis import _
from ibis import literal as L
from ibis.common.annotations import ValidationError
from ibis.common.exceptions import RelationError
from ibis.expr import api
from ibis.expr.types import Column, Table
from ibis.tests.expr.mocks import MockAlchemyBackend, MockBackend
from ibis.tests.util import assert_equal, assert_pickle_roundtrip


@pytest.fixture
def set_ops_schema_top():
    return [("key", "string"), ("value", "double")]


@pytest.fixture
def set_ops_schema_bottom():
    return [("key", "string"), ("key2", "string"), ("value", "double")]


@pytest.fixture
def setops_table_foo(set_ops_schema_top):
    return ibis.table(set_ops_schema_top, "foo")


@pytest.fixture
def setops_table_bar(set_ops_schema_top):
    return ibis.table(set_ops_schema_top, "bar")


@pytest.fixture
def setops_table_baz(set_ops_schema_bottom):
    return ibis.table(set_ops_schema_bottom, "baz")


@pytest.fixture
def setops_relation_error_message():
    return "Table schemas must be equal for set operations"


def test_empty_schema():
    table = api.table([], "foo")
    assert not table.schema()


def test_columns(con):
    t = con.table("alltypes")
    result = t.columns
    expected = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    assert result == expected


def test_view_new_relation(table):
    # For assisting with self-joins and other self-referential operations
    # where we need to be able to treat instances of the same Table as
    # semantically distinct
    #
    # This thing is not exactly a projection, since it has no semantic
    # meaning when it comes to execution
    tview = table.view()

    roots = an.find_immediate_parent_tables(tview.op())
    assert len(roots) == 1
    assert roots[0] is tview.op()


def test_getitem_column_select(table):
    for k in table.columns:
        col = table[k]

        # Make sure it's the right type
        assert isinstance(col, Column)


def test_table_tab_completion():
    table = ibis.table({"a": "int", "b": "int", "for": "int", "with spaces": "int"})
    # Only valid python identifiers in getattr tab completion
    attrs = set(dir(table))
    assert {"a", "b"}.issubset(attrs)
    assert {"for", "with spaces"}.isdisjoint(attrs)
    # All columns in getitem tab completion
    items = set(table._ipython_key_completions_())
    assert items.issuperset(table.columns)


def test_getitem_attribute(table):
    result = table.a
    assert_equal(result, table["a"])

    # Project and add a name that conflicts with a Table built-in
    # attribute
    view = table[[table, table["a"].name("schema")]]
    assert not isinstance(view.schema, Column)


def test_getitem_missing_column(table):
    with pytest.raises(com.IbisTypeError, match="oops"):
        table["oops"]


def test_getattr_missing_column(table):
    with pytest.raises(AttributeError, match="oops"):
        table.oops  # noqa: B018


def test_typo_method_name_recommendation(table):
    with pytest.raises(AttributeError, match="order_by"):
        table.sort("a")

    # Existing columns take precedence over raising an error
    # for a common method typo
    table2 = table.rename(sort="a")
    assert isinstance(table2.sort, Column)


def test_projection(table):
    cols = ["f", "a", "h"]

    proj = table[cols]
    assert isinstance(proj, Table)
    assert isinstance(proj.op(), ops.Selection)

    assert proj.schema().names == tuple(cols)
    for c in cols:
        expr = proj[c]
        assert isinstance(expr, type(table[c]))


def test_projection_no_list(table):
    expr = (table.f * 2).name("bar")
    result = table.select(expr)
    expected = table.select([expr])
    assert_equal(result, expected)


def test_projection_with_exprs(table):
    # unnamed expr to test
    mean_diff = (table["a"] - table["c"]).mean()

    col_exprs = [table["b"].log().name("log_b"), mean_diff.name("mean_diff")]

    proj = table[col_exprs + ["g"]]
    schema = proj.schema()
    assert schema.names == ("log_b", "mean_diff", "g")
    assert schema.types == (dt.double, dt.double, dt.string)

    # Test with unnamed expr
    proj = table.select(["g", table["a"] - table["c"]])
    schema = proj.schema()
    assert schema.names == ("g", "Subtract(a, c)")
    assert schema.types == (dt.string, dt.int64)


def test_projection_duplicate_names(table):
    with pytest.raises(com.IntegrityError):
        table.select([table.c, table.c])


def test_projection_invalid_root(table):
    schema1 = {"foo": "double", "bar": "int32"}

    left = api.table(schema1, name="foo")
    right = api.table(schema1, name="bar")

    exprs = [right["foo"], right["bar"]]
    with pytest.raises(RelationError):
        left.select(exprs)


def test_projection_with_star_expr(table):
    new_expr = (table["a"] * 5).name("bigger_a")

    t = table

    # it lives!
    proj = t[t, new_expr]
    repr(proj)

    ex_names = table.schema().names + ("bigger_a",)
    assert proj.schema().names == ex_names

    # cannot pass an invalid table expression
    t2 = t.aggregate([t["a"].sum().name("sum(a)")], by=["g"])
    with pytest.raises(RelationError):
        t[[t2]]
    # TODO: there may be some ways this can be invalid


def test_projection_convenient_syntax(table):
    proj = table[table, table["a"].name("foo")]
    proj2 = table[[table, table["a"].name("foo")]]
    assert_equal(proj, proj2)


def test_projection_mutate_analysis_bug(con):
    # GH #549

    t = con.table("airlines")

    filtered = t[t.depdelay.notnull()]
    leg = ibis.literal("-").join([t.origin, t.dest])
    mutated = filtered.mutate(leg=leg)

    # it works!
    mutated["year", "month", "day", "depdelay", "leg"]


def test_projection_self(table):
    result = table[table]
    expected = table.select(table)

    assert_equal(result, expected)


def test_projection_array_expr(table):
    result = table[table.a]
    expected = table[[table.a]]
    assert_equal(result, expected)


@pytest.mark.parametrize("empty", [list(), dict()])
def test_projection_no_expr(table, empty):
    with pytest.raises(com.IbisTypeError, match="must select at least one"):
        table.select(empty)


def test_projection_invalid_nested_list(table):
    errmsg = "must be coerceable to expressions"
    with pytest.raises(com.IbisTypeError, match=errmsg):
        table.select(["a", ["b"]])
    with pytest.raises(com.IbisTypeError, match=errmsg):
        table[["a", ["b"]]]
    with pytest.raises(com.IbisTypeError, match=errmsg):
        table["a", ["b"]]


def test_mutate(table):
    expr = table.mutate(
        [
            (table.a + 1).name("x1"),
            table.b.sum().name("x2"),
            (_.a + 2).name("x3"),
            lambda _: (_.a + 3).name("x4"),
            4,
            "five",
        ],
        kw1=(table.a + 6),
        kw2=table.b.sum(),
        kw3=(_.a + 7),
        kw4=lambda _: (_.a + 8),
        kw5=9,
        kw6="ten",
    )
    expected = table[
        table,
        (table.a + 1).name("x1"),
        table.b.sum().name("x2"),
        (table.a + 2).name("x3"),
        (table.a + 3).name("x4"),
        ibis.literal(4).name("4"),
        ibis.literal("five").name("'five'"),
        (table.a + 6).name("kw1"),
        table.b.sum().name("kw2"),
        (table.a + 7).name("kw3"),
        (table.a + 8).name("kw4"),
        ibis.literal(9).name("kw5"),
        ibis.literal("ten").name("kw6"),
    ]
    assert_equal(expr, expected)


def test_mutate_alter_existing_columns(table):
    new_f = table.f * 2
    foo = table.d * 2
    expr = table.mutate(f=new_f, foo=foo)

    expected = table[
        "a",
        "b",
        "c",
        "d",
        "e",
        new_f.name("f"),
        "g",
        "h",
        "i",
        "j",
        "k",
        foo.name("foo"),
    ]

    assert_equal(expr, expected)


def test_replace_column():
    tb = api.table([("a", "int32"), ("b", "double"), ("c", "string")])

    expr = tb.b.cast("int32")
    tb2 = tb.mutate(b=expr)
    expected = tb[tb.a, expr.name("b"), tb.c]

    assert_equal(tb2, expected)


def test_filter_no_list(table):
    pred = table.a > 5

    result = table.filter(pred)
    expected = table[pred]
    assert_equal(result, expected)


def test_add_predicate(table):
    pred = table["a"] > 5
    result = table[pred]
    assert isinstance(result.op(), ops.Selection)


def test_invalid_predicate(table, schema):
    # a lookalike
    table2 = api.table(schema, name="bar")
    predicate = table2.a > 5
    with pytest.raises(RelationError):
        table.filter(predicate)


def test_add_predicate_coalesce(table):
    # Successive predicates get combined into one rather than nesting. This
    # is mainly to enhance readability since we could handle this during
    # expression evaluation anyway.
    pred1 = table["a"] > 5
    pred2 = table["b"] > 0

    result = table[pred1][pred2]
    expected = table.filter([pred1, pred2])
    assert_equal(result, expected)

    # 59, if we are not careful, we can obtain broken refs
    subset = table[pred1]
    result = subset.filter([subset["b"] > 0])
    assert_equal(result, expected)


def test_repr_same_but_distinct_objects(con):
    t = con.table("test1")
    t_copy = con.table("test1")
    table2 = t[t_copy["f"] > 0]

    result = repr(table2)
    assert result.count("DatabaseTable") == 1


def test_filter_fusion_distinct_table_objects(con):
    t = con.table("test1")
    tt = con.table("test1")

    expr = t[t.f > 0][t.c > 0]
    expr2 = t[t.f > 0][tt.c > 0]
    expr3 = t[tt.f > 0][tt.c > 0]
    expr4 = t[tt.f > 0][t.c > 0]

    assert_equal(expr, expr2)
    assert repr(expr) == repr(expr2)
    assert_equal(expr, expr3)
    assert_equal(expr, expr4)


def test_relabel():
    table = api.table({"x": "int32", "y": "string", "z": "double"})

    # Using a mapping
    with pytest.warns(FutureWarning, match="Table.rename"):
        res = table.relabel({"x": "x_1", "y": "y_1"}).schema()
    sol = sch.schema({"x_1": "int32", "y_1": "string", "z": "double"})
    assert_equal(res, sol)

    # Using a function
    with pytest.warns(FutureWarning, match="Table.rename"):
        res = table.relabel(lambda x: None if x == "z" else f"{x}_1").schema()
    assert_equal(res, sol)

    # Using a format string
    with pytest.warns(FutureWarning, match="Table.rename"):
        res = table.relabel("_{name}_")
        sol = table.relabel({"x": "_x_", "y": "_y_", "z": "_z_"})
    assert_equal(res, sol)

    # Mapping with unknown columns errors
    with pytest.raises(com.IbisTypeError, match="'missing' is not found in table"):
        with pytest.warns(FutureWarning, match="Table.rename"):
            table.relabel({"missing": "oops"})


def test_rename():
    table = api.table({"x": "int32", "y": "string", "z": "double"})
    sol = sch.schema({"x_1": "int32", "y_1": "string", "z": "double"})

    # Using kwargs
    res = table.rename(x_1="x", y_1="y").schema()
    assert_equal(res, sol)

    # Using a mapping
    res = table.rename({"x_1": "x", "y_1": "y"}).schema()
    assert_equal(res, sol)

    # Using a mix
    res = table.rename({"x_1": "x"}, y_1="y").schema()
    assert_equal(res, sol)


def test_rename_function():
    table = api.table({"x": "int32", "y": "string", "z": "double"})

    res = table.rename(lambda x: None if x == "z" else f"{x}_1").schema()
    sol = sch.schema({"x_1": "int32", "y_1": "string", "z": "double"})
    assert_equal(res, sol)

    # Explicit rename takes precedence
    res = table.rename(lambda x: f"{x}_1", z_2="z").schema()
    sol = sch.schema({"x_1": "int32", "y_1": "string", "z_2": "double"})
    assert_equal(res, sol)


def test_rename_format_string():
    t = ibis.table({"x": "int", "y": "int", "z": "int"})

    res = t.rename("_{name}_")
    sol = t.rename({"_x_": "x", "_y_": "y", "_z_": "z"})
    assert_equal(res, sol)

    with pytest.raises(ValueError, match="Format strings must"):
        t.rename("no format string parameter")

    with pytest.raises(ValueError, match="Format strings must"):
        t.rename("{unknown} format string parameter")


def test_rename_snake_case():
    cases = [
        ("cola", "cola"),
        ("col_b", "ColB"),
        ("col_c", "colC"),
        ("col_d", "col-d"),
        ("col_e", "col_e"),
        ("column_f", " Column F "),
        ("column_g_with_hyphens", "Column G-with-hyphens"),
        ("col_h_notcamelcase", "Col H notCamelCase"),
    ]
    t = ibis.table({c: "int" for _, c in cases})
    res = t.rename("snake_case")
    sol = t.rename(dict(cases))
    assert_equal(res, sol)


def test_rename_all_caps():
    cases = [
        ("COLA", "cola"),
        ("COL_B", "ColB"),
        ("COL_C", "colC"),
        ("COL_D", "col-d"),
        ("COL_E", "col_e"),
        ("COLUMN_F", " Column F "),
        ("COLUMN_G_WITH_HYPHENS", "Column G-with-hyphens"),
        ("COL_H_NOTCAMELCASE", "Col H notCamelCase"),
    ]
    t = ibis.table({c: "int" for _, c in cases})
    res = t.rename("ALL_CAPS")
    sol = t.rename(dict(cases))
    assert_equal(res, sol)


def test_limit(table):
    limited = table.limit(10, offset=5)
    assert limited.op().n == 10
    assert limited.op().offset == 5


def test_order_by(table):
    result = table.order_by(["f"]).op()

    sort_key = result.sort_keys[0]

    assert_equal(sort_key.expr, table.f.op())
    assert sort_key.ascending

    # non-list input. per #150
    result2 = table.order_by("f").op()
    assert_equal(result, result2)

    key2 = result2.sort_keys[0]
    assert key2.descending is False


def test_order_by_desc_deferred_sort_key(table):
    result = table.group_by("g").size().order_by(ibis._[1].desc())

    tmp = table.group_by("g").size()
    expected = tmp.order_by(ibis.desc(tmp[1]))

    assert_equal(result, expected)


def test_order_by_asc_deferred_sort_key(table):
    result = table.group_by("g").size().order_by(ibis._[1])

    tmp = table.group_by("g").size()
    expected = tmp.order_by(tmp[1])
    expected2 = tmp.order_by(ibis.asc(tmp[1]))

    assert_equal(result, expected)
    assert_equal(result, expected2)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        param(ibis.NA, ibis.NA.op(), id="na"),
        param(ibis.random(), ibis.random().op(), id="random"),
        param(1.0, L(1.0).op(), id="float"),
        param(L("a"), L("a").op(), id="string"),
        param(L([1, 2, 3]), L([1, 2, 3]).op(), id="array"),
    ],
)
def test_order_by_scalar(table, key, expected):
    result = table.order_by(key)
    assert result.op().sort_keys == (ops.SortKey(expected),)


@pytest.mark.parametrize(
    ("key", "exc_type"),
    [
        ("bogus", com.IbisTypeError),
        (("bogus", False), com.IbisTypeError),
        (ibis.desc("bogus"), com.IbisTypeError),
        (1000, IndexError),
        ((1000, False), IndexError),
        (_.bogus, AttributeError),
        (_.bogus.desc(), AttributeError),
    ],
)
@pytest.mark.parametrize(
    "expr_func",
    [
        param(lambda t: t, id="table"),
        param(lambda t: t.select("a", "b"), id="selection"),
        param(lambda t: t.group_by("a").agg(new=_.b.sum()), id="aggregation"),
    ],
)
def test_order_by_nonexistent_column_errors(table, expr_func, key, exc_type):
    # `order_by` is implemented on a few different operations, we check them
    # all in turn here.
    expr = expr_func(table)
    with pytest.raises(exc_type):
        expr.order_by(key)


def test_slice(table):
    expr1 = table[:5]
    expr2 = table[:5:1]
    expr3 = table[5:]
    assert_equal(expr1, table.limit(5))
    assert_equal(expr1, expr2)
    assert_equal(expr3, table.limit(None, offset=5))

    expr1 = table[2:7]
    expr2 = table[2:7:1]
    expr3 = table[2::1]
    assert_equal(expr1, table.limit(5, offset=2))
    assert_equal(expr1, expr2)
    assert_equal(expr3, table.limit(None, offset=2))


@pytest.mark.parametrize("step", [-1, 0, 2])
def test_invalid_slice(table, step):
    with pytest.raises(ValueError):
        table[:5:step]


def test_table_count(table):
    result = table.count()
    assert isinstance(result, ir.IntegerScalar)
    assert isinstance(result.op(), ops.CountStar)


def test_len_raises_expression_error(table):
    with pytest.raises(com.ExpressionError):
        len(table)


def test_sum_expr_basics(table, int_col):
    # Impala gives bigint for all integer types
    result = table[int_col].sum()
    assert isinstance(result, ir.IntegerScalar)
    assert isinstance(result.op(), ops.Sum)


def test_sum_expr_basics_floats(table, float_col):
    # Impala gives double for all floating point types
    result = table[float_col].sum()
    assert isinstance(result, ir.FloatingScalar)
    assert isinstance(result.op(), ops.Sum)


def test_mean_expr_basics(table, numeric_col):
    result = table[numeric_col].mean()
    assert isinstance(result, ir.FloatingScalar)
    assert isinstance(result.op(), ops.Mean)


def test_aggregate_no_keys(table):
    metrics = [
        table["a"].sum().name("sum(a)"),
        table["c"].mean().name("mean(c)"),
    ]

    # A Table, which in SQL at least will yield a table with a single
    # row
    result = table.aggregate(metrics)
    assert isinstance(result, Table)


def test_aggregate_keys_basic(table):
    metrics = [
        table["a"].sum().name("sum(a)"),
        table["c"].mean().name("mean(c)"),
    ]

    # A Table, which in SQL at least will yield a table with a single
    # row
    result = table.aggregate(metrics, by=["g"])
    assert isinstance(result, Table)

    # it works!
    repr(result)


def test_aggregate_non_list_inputs(table):
    # per #150
    metric = table.f.sum().name("total")
    by = "g"
    having = table.c.sum() > 10

    result = table.aggregate(metric, by=by, having=having)
    expected = table.aggregate([metric], by=[by], having=[having])
    assert_equal(result, expected)


def test_aggregate_keywords(table):
    t = table

    expr = t.aggregate(foo=t.f.sum(), bar=lambda x: x.f.mean(), by="g")
    expr2 = t.group_by("g").aggregate(foo=t.f.sum(), bar=lambda x: x.f.mean())
    expected = t.aggregate([t.f.sum().name("foo"), t.f.mean().name("bar")], by="g")

    assert_equal(expr, expected)
    assert_equal(expr2, expected)


def test_filter_aggregate_pushdown_predicate(table):
    # In the case where we want to add a predicate to an aggregate
    # expression after the fact, rather than having to backpedal and add it
    # before calling aggregate.
    #
    # TODO (design decision): This could happen automatically when adding a
    # predicate originating from the same root table; if an expression is
    # created from field references from the aggregated table then it
    # becomes a filter predicate applied on top of a view

    pred = table.f > 0
    metrics = [table.a.sum().name("total")]
    agged = table.aggregate(metrics, by=["g"])
    filtered = agged.filter([pred])
    expected = table[pred].aggregate(metrics, by=["g"])
    assert_equal(filtered, expected)


def test_filter_on_literal_then_aggregate(table):
    # Mostly just a smoketest, this used to error on construction
    expr = table.filter(ibis.literal(True)).agg(lambda t: t.a.sum().name("total"))
    assert expr.columns == ["total"]


@pytest.mark.parametrize(
    "case_fn",
    [
        param(lambda t: t.f.sum(), id="non_boolean"),
        param(lambda t: t.f > 2, id="non_scalar"),
    ],
)
def test_aggregate_post_predicate(table, case_fn):
    # Test invalid having clause
    metrics = [table.f.sum().name("total")]
    by = ["g"]
    having = [case_fn(table)]

    with pytest.raises(ValidationError):
        table.aggregate(metrics, by=by, having=having)


def test_group_by_having_api(table):
    # #154, add a HAVING post-predicate in a composable way
    metric = table.f.sum().name("foo")
    postp = table.d.mean() > 1

    expr = table.group_by("g").having(postp).aggregate(metric)

    expected = table.aggregate(metric, by="g", having=postp)
    assert_equal(expr, expected)


def test_group_by_kwargs(table):
    t = table
    expr = t.group_by(["f", t.h], z="g", z2=t.d).aggregate(t.d.mean().name("foo"))
    expected = t.group_by(["f", t.h, t.g.name("z"), t.d.name("z2")]).aggregate(
        t.d.mean().name("foo")
    )
    assert_equal(expr, expected)


def test_compound_aggregate_expr(table):
    # See ibis #24
    compound_expr = (table["a"].sum() / table["a"].mean()).name("foo")

    # Validates internally
    table.aggregate([compound_expr])


def test_groupby_convenience(table):
    metrics = [table.f.sum().name("total")]

    expr = table.group_by("g").aggregate(metrics)
    expected = table.aggregate(metrics, by=["g"])
    assert_equal(expr, expected)

    group_expr = table.g.cast("double").name("g")
    expr = table.group_by(group_expr).aggregate(metrics)
    expected = table.aggregate(metrics, by=[group_expr])
    assert_equal(expr, expected)


def test_group_by_count_size(table):
    # #148, convenience for interactive use, and so forth
    result1 = table.group_by("g").size()
    result2 = table.group_by("g").count()

    expected = table.group_by("g").aggregate(table.count())

    assert_equal(result1, expected)
    assert_equal(result2, expected)


def test_group_by_column_select_api(table):
    grouped = table.group_by("g")

    result = grouped.f.sum()
    expected = grouped.aggregate(table.f.sum().name("sum(f)"))
    assert_equal(result, expected)

    supported_functions = ["sum", "mean", "count", "size", "max", "min"]

    # make sure they all work
    for fn in supported_functions:
        getattr(grouped.f, fn)()


def test_value_counts_convenience(table):
    # #152
    result = table.g.value_counts()
    expected = table.select("g").group_by("g").aggregate(g_count=lambda t: t.count())

    assert_equal(result, expected)


def test_isin_value_counts(table):
    # #157, this code path was untested before
    bool_clause = table.g.notin(["1", "4", "7"])
    # it works!
    bool_clause.name("notin").value_counts()


def test_value_counts_unnamed_expr(con):
    nation = con.table("tpch_nation")

    expr = nation.n_name.lower().value_counts()
    expected = nation.n_name.lower().name("Lowercase(n_name)").value_counts()
    assert_equal(expr, expected)


def test_aggregate_unnamed_expr(con):
    nation = con.table("tpch_nation")
    expr = nation.n_name.lower().left(1)

    agg = nation.group_by(expr).aggregate(nation.count().name("metric"))
    schema = agg.schema()
    assert schema.names == ("Substring(Lowercase(n_name), 0, 1)", "metric")
    assert schema.types == (dt.string, dt.int64)


def test_join_no_predicate_list(con):
    region = con.table("tpch_region")
    nation = con.table("tpch_nation")

    pred = region.r_regionkey == nation.n_regionkey
    joined = region.inner_join(nation, pred)
    expected = region.inner_join(nation, [pred])
    assert_equal(joined, expected)


def test_join_deferred(con):
    region = con.table("tpch_region")
    nation = con.table("tpch_nation")
    res = region.join(nation, _.r_regionkey == nation.n_regionkey)
    exp = region.join(nation, region.r_regionkey == nation.n_regionkey)
    assert_equal(res, exp)


def test_asof_join():
    left = ibis.table([("time", "int32"), ("value", "double")])
    right = ibis.table([("time", "int32"), ("value2", "double")])
    joined = api.asof_join(left, right, "time")

    assert joined.columns == [
        "time",
        "value",
        "time_right",
        "value2",
    ]
    pred = joined.op().table.predicates[0]
    assert pred.left.name == pred.right.name == "time"


def test_asof_join_with_by():
    left = ibis.table([("time", "int32"), ("key", "int32"), ("value", "double")])
    right = ibis.table([("time", "int32"), ("key", "int32"), ("value2", "double")])
    joined = api.asof_join(left, right, "time", by="key")
    assert joined.columns == [
        "time",
        "key",
        "value",
        "time_right",
        "key_right",
        "value2",
    ]
    by = joined.op().table.by[0]
    assert by.left.name == by.right.name == "key"


@pytest.mark.parametrize(
    ("ibis_interval", "timedelta_interval"),
    [
        [ibis.interval(days=2), pd.Timedelta("2 days")],
        [ibis.interval(days=2), datetime.timedelta(days=2)],
        [ibis.interval(hours=5), pd.Timedelta("5 hours")],
        [ibis.interval(hours=5), datetime.timedelta(hours=5)],
        [ibis.interval(minutes=7), pd.Timedelta("7 minutes")],
        [ibis.interval(minutes=7), datetime.timedelta(minutes=7)],
        [ibis.interval(seconds=9), pd.Timedelta("9 seconds")],
        [ibis.interval(seconds=9), datetime.timedelta(seconds=9)],
        [ibis.interval(milliseconds=11), pd.Timedelta("11 milliseconds")],
        [ibis.interval(milliseconds=11), datetime.timedelta(milliseconds=11)],
        [ibis.interval(microseconds=15), pd.Timedelta("15 microseconds")],
        [ibis.interval(microseconds=15), datetime.timedelta(microseconds=15)],
        [ibis.interval(nanoseconds=17), pd.Timedelta("17 nanoseconds")],
    ],
)
def test_asof_join_with_tolerance(ibis_interval, timedelta_interval):
    left = ibis.table([("time", "int32"), ("key", "int32"), ("value", "double")])
    right = ibis.table([("time", "int32"), ("key", "int32"), ("value2", "double")])

    joined = api.asof_join(left, right, "time", tolerance=ibis_interval).op()
    tolerance = joined.table.tolerance
    assert_equal(tolerance, ibis_interval.op())

    joined = api.asof_join(left, right, "time", tolerance=timedelta_interval).op()
    tolerance = joined.table.tolerance
    assert isinstance(tolerance.to_expr(), ir.IntervalScalar)
    assert isinstance(tolerance, ops.Literal)


def test_equijoin_schema_merge():
    table1 = ibis.table([("key1", "string"), ("value1", "double")])
    table2 = ibis.table([("key2", "string"), ("stuff", "int32")])

    pred = table1["key1"] == table2["key2"]
    join_types = ["inner_join", "left_join", "outer_join"]

    ex_schema = ibis.schema(
        names=["key1", "value1", "key2", "stuff"],
        types=["string", "double", "string", "int32"],
    )

    for fname in join_types:
        f = getattr(table1, fname)
        joined = f(table2, [pred])
        assert_equal(joined.schema(), ex_schema)


def test_join_combo_with_projection(table):
    # Test a case where there is column name overlap, but the projection
    # passed makes it a non-issue. Highly relevant with self-joins
    #
    # For example, where left/right have some field names in common:
    # SELECT left.*, right.a, right.b
    # FROM left join right on left.key = right.key
    t = table
    t2 = t.mutate(foo=t.f * 2, bar=t.f * 4)

    # this works
    joined = t.left_join(t2, [t["g"] == t2["g"]])
    proj = joined.select([t, t2["foo"], t2["bar"]])
    repr(proj)


def test_join_getitem_projection(con):
    region = con.table("tpch_region")
    nation = con.table("tpch_nation")

    pred = region.r_regionkey == nation.n_regionkey
    joined = region.inner_join(nation, pred)

    result = joined[nation]
    expected = joined.select(nation)
    assert_equal(result, expected)


def test_self_join(table):
    # Self-joins are problematic with this design because column
    # expressions may reference either the left or right  For example:
    #
    # SELECT left.key, sum(left.value - right.value) as total_deltas
    # FROM table left
    #  INNER JOIN table right
    #    ON left.current_period = right.previous_period + 1
    # GROUP BY 1
    #
    # One way around the self-join issue is to force the user to add
    # prefixes to the joined fields, then project using those. Not that
    # satisfying, though.
    left = table
    right = table.view()
    metric = (left["a"] - right["b"]).mean().name("metric")

    joined = left.inner_join(right, [right["g"] == left["g"]])

    # Project out left table schema
    proj = joined[[left]]
    assert_equal(proj.schema(), left.schema())

    # Try aggregating on top of joined
    aggregated = joined.aggregate([metric], by=[left["g"]])
    ex_schema = api.Schema({"g": "string", "metric": "double"})
    assert_equal(aggregated.schema(), ex_schema)


def test_self_join_no_view_convenience(table):
    # #165, self joins ought to be possible when the user specifies the
    # column names to join on rather than referentially-valid expressions

    result = table.join(table, [("g", "g")])
    expected_cols = list(table.columns)
    expected_cols.extend(f"{c}_right" for c in table.columns if c != "g")
    assert result.columns == expected_cols


def test_join_reference_bug(con):
    # GH#403
    orders = con.table("tpch_orders")
    customer = con.table("tpch_customer")
    lineitem = con.table("tpch_lineitem")

    items = orders.join(lineitem, orders.o_orderkey == lineitem.l_orderkey)[
        lineitem, orders.o_custkey, orders.o_orderpriority
    ].join(customer, [("o_custkey", "c_custkey")])
    items["o_orderpriority"].value_counts()


def test_join_project_after(table):
    # e.g.
    #
    # SELECT L.foo, L.bar, R.baz, R.qux
    # FROM table1 L
    #   INNER JOIN table2 R
    #     ON L.key = R.key
    #
    # or
    #
    # SELECT L.*, R.baz
    # ...
    #
    # The default for a join is selecting all fields if possible
    table1 = ibis.table([("key1", "string"), ("value1", "double")])
    table2 = ibis.table([("key2", "string"), ("stuff", "int32")])

    pred = table1["key1"] == table2["key2"]

    joined = table1.left_join(table2, [pred])
    projected = joined.select([table1, table2["stuff"]])
    assert projected.schema().names == ("key1", "value1", "stuff")

    projected = joined.select([table2, table1["key1"]])
    assert projected.schema().names == ("key2", "stuff", "key1")


def test_semi_join_schema(table):
    # A left semi join discards the schema of the right table
    table1 = ibis.table([("key1", "string"), ("value1", "double")])
    table2 = ibis.table([("key2", "string"), ("stuff", "double")])

    pred = table1["key1"] == table2["key2"]
    semi_joined = table1.semi_join(table2, [pred])

    result_schema = semi_joined.schema()
    assert_equal(result_schema, table1.schema())


def test_cross_join(table):
    metrics = [
        table["a"].sum().name("sum_a"),
        table["b"].mean().name("mean_b"),
    ]
    scalar_aggs = table.aggregate(metrics)

    joined = table.cross_join(scalar_aggs)
    agg_schema = api.Schema({"sum_a": "int64", "mean_b": "double"})
    ex_schema = table.schema() | agg_schema
    assert_equal(joined.schema(), ex_schema)


def test_cross_join_multiple(table):
    a = table["a", "b", "c"]
    b = table["d", "e"]
    c = table["f", "h"]

    joined = ibis.cross_join(a, b, c)
    expected = a.cross_join(b.cross_join(c))
    assert joined.equals(expected)


def test_filter_join():
    table1 = ibis.table({"key1": "string", "key2": "string", "value1": "double"})
    table2 = ibis.table({"key3": "string", "value2": "double"})

    # It works!
    joined = table1.inner_join(table2, [table1["key1"] == table2["key3"]])
    filtered = joined.filter([table1.value1 > 0])
    repr(filtered)


def test_inner_join_overlapping_column_names():
    t1 = ibis.table([("foo", "string"), ("bar", "string"), ("value1", "double")])
    t2 = ibis.table([("foo", "string"), ("bar", "string"), ("value2", "double")])

    joined = t1.join(t2, "foo")
    expected = t1.join(t2, t1.foo == t2.foo)
    assert_equal(joined, expected)
    assert joined.columns == ["foo", "bar", "value1", "bar_right", "value2"]

    joined = t1.join(t2, ["foo", "bar"])
    expected = t1.join(t2, [t1.foo == t2.foo, t1.bar == t2.bar])
    assert_equal(joined, expected)
    assert joined.columns == ["foo", "bar", "value1", "value2"]

    # Equality predicates don't have same name, need to rename
    joined = t1.join(t2, t1.foo == t2.bar)
    assert joined.columns == [
        "foo",
        "bar",
        "value1",
        "foo_right",
        "bar_right",
        "value2",
    ]

    # Not all predicates are equality, still need to rename
    joined = t1.join(t2, ["foo", t1.value1 < t2.value2])
    assert joined.columns == [
        "foo",
        "bar",
        "value1",
        "foo_right",
        "bar_right",
        "value2",
    ]


@pytest.mark.parametrize(
    "key_maker",
    [
        lambda t1, t2: t1.foo_id == t2.foo_id,
        lambda t1, t2: [("foo_id", "foo_id")],
        lambda t1, t2: [(t1.foo_id, t2.foo_id)],
        lambda t1, t2: [(_.foo_id, _.foo_id)],
        lambda t1, t2: [(t1.foo_id, _.foo_id)],
        lambda t1, t2: [(2, 0)],  # foo_id is 2nd in t1, 0th in t2
        lambda t1, t2: [(lambda t: t.foo_id, lambda t: t.foo_id)],
    ],
)
def test_join_key_alternatives(con, key_maker):
    t1 = con.table("star1")
    t2 = con.table("star2")
    expected = t1.inner_join(t2, [t1.foo_id == t2.foo_id])
    key = key_maker(t1, t2)
    joined = t1.inner_join(t2, key)
    assert_equal(joined, expected)


@pytest.mark.parametrize(
    "key,error",
    [
        ([("foo_id", "foo_id", "foo_id")], com.ExpressionError),
        ([(s.c("foo_id"), s.c("foo_id"))], ValueError),
    ],
)
def test_join_key_invalid(con, key, error):
    t1 = con.table("star1")
    t2 = con.table("star2")
    with pytest.raises(error):
        t1.inner_join(t2, key)


def test_join_invalid_refs(con):
    t1 = con.table("star1")
    t2 = con.table("star2")
    t3 = con.table("star3")

    predicate = t1.bar_id == t3.bar_id
    with pytest.raises(com.RelationError):
        t1.inner_join(t2, [predicate])


def test_join_invalid_expr_type(con):
    left = con.table("star1")
    invalid_right = left.foo_id
    join_key = ["bar_id"]

    with pytest.raises(ValidationError):
        left.inner_join(invalid_right, join_key)


def test_join_non_boolean_expr(con):
    t1 = con.table("star1")
    t2 = con.table("star2")

    # oops
    predicate = t1.f * t2.value1
    with pytest.raises(com.ExpressionError):
        t1.inner_join(t2, [predicate])


def test_unravel_compound_equijoin(table):
    t1 = ibis.table(
        [
            ("key1", "string"),
            ("key2", "string"),
            ("key3", "string"),
            ("value1", "double"),
        ],
        "foo_table",
    )

    t2 = ibis.table(
        [
            ("key1", "string"),
            ("key2", "string"),
            ("key3", "string"),
            ("value2", "double"),
        ],
        "bar_table",
    )

    p1 = t1.key1 == t2.key1
    p2 = t1.key2 == t2.key2
    p3 = t1.key3 == t2.key3

    joined = t1.inner_join(t2, [p1 & p2 & p3])
    expected = t1.inner_join(t2, [p1, p2, p3])
    assert_equal(joined, expected)


def test_union(
    setops_table_foo,
    setops_table_bar,
    setops_table_baz,
    setops_relation_error_message,
):
    result = setops_table_foo.union(setops_table_bar)
    assert isinstance(result.op().table, ops.Union)
    assert not result.op().table.distinct

    result = setops_table_foo.union(setops_table_bar, distinct=True)
    assert result.op().table.distinct

    with pytest.raises(RelationError, match=setops_relation_error_message):
        setops_table_foo.union(setops_table_baz)


def test_intersection(
    setops_table_foo,
    setops_table_bar,
    setops_table_baz,
    setops_relation_error_message,
):
    result = setops_table_foo.intersect(setops_table_bar)
    assert isinstance(result.op().table, ops.Intersection)

    with pytest.raises(RelationError, match=setops_relation_error_message):
        setops_table_foo.intersect(setops_table_baz)


def test_difference(
    setops_table_foo,
    setops_table_bar,
    setops_table_baz,
    setops_relation_error_message,
):
    result = setops_table_foo.difference(setops_table_bar)
    assert isinstance(result.op().table, ops.Difference)

    with pytest.raises(RelationError, match=setops_relation_error_message):
        setops_table_foo.difference(setops_table_baz)


def test_column_ref_on_projection_rename(con):
    region = con.table("tpch_region")
    nation = con.table("tpch_nation")
    customer = con.table("tpch_customer")

    joined = region.inner_join(
        nation, [region.r_regionkey == nation.n_regionkey]
    ).inner_join(customer, [customer.c_nationkey == nation.n_nationkey])

    proj_exprs = [
        customer,
        nation.n_name.name("nation"),
        region.r_name.name("region"),
    ]
    joined = joined.select(proj_exprs)

    metrics = [joined.c_acctbal.sum().name("metric")]

    # it works!
    joined.aggregate(metrics, by=["region"])


@pytest.fixture
def t1():
    return ibis.table(
        [("key1", "string"), ("key2", "string"), ("value1", "double")], "foo"
    )


@pytest.fixture
def t2():
    return ibis.table([("key1", "string"), ("key2", "string")], "bar")


def test_unresolved_existence_predicate(t1, t2):
    expr = (t1.key1 == t2.key1).any()
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.UnresolvedExistsSubquery)


def test_resolve_existence_predicate(t1, t2):
    expr = t1[(t1.key1 == t2.key1).any()]
    op = expr.op()
    assert isinstance(op, ops.Selection)

    pred = op.predicates[0].to_expr()
    assert isinstance(pred.op(), ops.ExistsSubquery)


def test_aggregate_metrics(table):
    functions = [
        lambda x: x.e.sum().name("esum"),
        lambda x: x.f.sum().name("fsum"),
    ]
    exprs = [table.e.sum().name("esum"), table.f.sum().name("fsum")]

    result = table.aggregate(functions[0])
    expected = table.aggregate(exprs[0])
    assert_equal(result, expected)

    result = table.aggregate(functions)
    expected = table.aggregate(exprs)
    assert_equal(result, expected)


def test_group_by_keys(table):
    m = table.mutate(foo=table.f * 2, bar=table.e / 2)

    expr = m.group_by(lambda x: x.foo).size()
    expected = m.group_by("foo").size()
    assert_equal(expr, expected)

    expr = m.group_by([lambda x: x.foo, lambda x: x.bar]).size()
    expected = m.group_by(["foo", "bar"]).size()
    assert_equal(expr, expected)


def test_having(table):
    m = table.mutate(foo=table.f * 2, bar=table.e / 2)

    expr = m.group_by("foo").having(lambda x: x.foo.sum() > 10).size()
    expected = m.group_by("foo").having(m.foo.sum() > 10).size()

    assert_equal(expr, expected)


def test_filter(table):
    m = table.mutate(foo=table.f * 2, bar=table.e / 2)

    result = m.filter(lambda x: x.foo > 10)
    result2 = m[lambda x: x.foo > 10]
    expected = m[m.foo > 10]

    assert_equal(result, expected)
    assert_equal(result2, expected)

    result = m.filter([lambda x: x.foo > 10, lambda x: x.bar < 0])
    expected = m.filter([m.foo > 10, m.bar < 0])
    assert_equal(result, expected)


def test_order_by2(table):
    m = table.mutate(foo=table.e + table.f)

    result = m.order_by(lambda x: -x.foo)
    expected = m.order_by(-m.foo)
    assert_equal(result, expected)

    result = m.order_by(lambda x: ibis.desc(x.foo))
    expected = m.order_by(ibis.desc("foo"))
    assert_equal(result, expected)

    result = m.order_by(ibis.desc(lambda x: x.foo))
    expected = m.order_by(ibis.desc("foo"))
    assert_equal(result, expected)

    result = m.order_by(ibis.asc(lambda x: x.foo))
    expected = m.order_by("foo")
    assert_equal(result, expected)


def test_projection2(table):
    m = table.mutate(foo=table.f * 2)

    def f(x):
        return (x.foo * 2).name("bar")

    result = m.select([f, "f"])
    result2 = m[f, "f"]
    expected = m.select([f(m), "f"])
    assert_equal(result, expected)
    assert_equal(result2, expected)


def test_mutate2(table):
    m = table.mutate(foo=table.f * 2)

    def g(x):
        return x.foo * 2

    def h(x):
        return x.bar * 2

    result = m.mutate(bar=g).mutate(baz=h)

    m2 = m.mutate(bar=g(m))
    expected = m2.mutate(baz=h(m2))

    assert_equal(result, expected)


def test_groupby_mutate(table):
    t = table

    g = t.group_by("g").order_by("f")
    expr = g.mutate(foo=lambda x: x.f.lag(), bar=lambda x: x.f.rank())
    expected = g.mutate(foo=t.f.lag(), bar=t.f.rank())

    assert_equal(expr, expected)


def test_groupby_projection(table):
    t = table

    g = t.group_by("g").order_by("f")
    expr = g.select([lambda x: x.f.lag().name("foo"), lambda x: x.f.rank().name("bar")])
    expected = g.select([t.f.lag().name("foo"), t.f.rank().name("bar")])

    assert_equal(expr, expected)


def test_pickle_table_expr():
    schema = [("time", "timestamp"), ("key", "string"), ("value", "double")]
    t0 = ibis.table(schema, name="t0")
    raw = pickle.dumps(t0, protocol=2)
    t1 = pickle.loads(raw)
    assert t1.equals(t0)


def test_pickle_table_node(table):
    n0 = table.op()
    assert_pickle_roundtrip(n0)


def test_pickle_projection_node(table):
    m = table.mutate(foo=table.f * 2)

    def f(x):
        return (x.foo * 2).name("bar")

    node = m.select([f, "f"]).op()

    assert_pickle_roundtrip(node)


def test_pickle_group_by(table):
    m = table.mutate(foo=table.f * 2, bar=table.e / 2)
    expr = m.group_by(lambda x: x.foo).size()
    node = expr.op()

    assert_pickle_roundtrip(node)


def test_pickle_asof_join():
    left = ibis.table([("time", "int32"), ("value", "double")])
    right = ibis.table([("time", "int32"), ("value2", "double")])
    joined = api.asof_join(left, right, "time")
    node = joined.op()

    assert_pickle_roundtrip(node)


def test_group_by_key_function():
    t = ibis.table([("a", "timestamp"), ("b", "string"), ("c", "double")])
    expr = t.group_by(new_key=lambda t: t.b.length()).aggregate(foo=t.c.mean())
    assert expr.columns == ["new_key", "foo"]


def test_group_by_no_keys():
    t = ibis.table([("a", "timestamp"), ("b", "string"), ("c", "double")])

    with pytest.raises(com.IbisInputError):
        t.group_by(s.startswith("x")).aggregate(foo=t.c.mean())


def test_unbound_table_name():
    t = ibis.table([("a", "timestamp")])
    name = t.op().name
    match = re.match(r"^unbound_table_\d+$", name)
    assert match is not None


class MyTable:
    a: int
    b: str
    c: list[float]


def test_unbound_table_using_class_definition():
    expected_schema = ibis.schema({"a": "int64", "b": "string", "c": "array<double>"})

    t1 = ibis.table(MyTable)
    t2 = ibis.table(MyTable, name="MyNamedTable")

    cases = {t1: "MyTable", t2: "MyNamedTable"}
    for t, name in cases.items():
        assert isinstance(t, ir.TableExpr)
        assert isinstance(t.op(), ops.UnboundTable)
        assert t.schema() == expected_schema
        assert t.get_name() == name


def test_mutate_chain():
    one = ibis.table([("a", "string"), ("b", "string")], name="t")
    two = one.mutate(b=lambda t: t.b.fillna("Short Term"))
    three = two.mutate(a=lambda t: t.a.fillna("Short Term"))
    a, b = three.op().selections

    # we can't fuse these correctly yet
    assert isinstance(a, ops.Alias)
    assert isinstance(a.arg, ops.Coalesce)
    assert isinstance(b, ops.TableColumn)

    expr = b.table.selections[1]
    assert isinstance(expr, ops.Alias)
    assert isinstance(expr.arg, ops.Coalesce)


# TODO(kszucs): move this test case to ibis/tests/sql since it requires the
# sql backend to be executed
def test_multiple_dbcon():
    """Expr from multiple connections to same DB should be compatible."""
    con1 = MockBackend()
    con2 = MockBackend()

    con1.table("alltypes").union(con2.table("alltypes")).execute()


def test_multiple_db_different_backends():
    con1 = MockBackend()
    con2 = MockAlchemyBackend()

    backend1_table = con1.table("alltypes")
    backend2_table = con2.table("alltypes")

    expr = backend1_table.union(backend2_table)
    with pytest.raises(com.IbisError, match="Multiple backends"):
        expr.compile()


def test_merge_as_of_allows_overlapping_columns():
    # GH3295
    table = ibis.table(
        [
            ("field", "string"),
            ("value", "float64"),
            ("timestamp_received", "timestamp"),
        ],
        name="t",
    )

    signal_one = table[
        table["field"].contains("signal_one") & table["field"].contains("current")
    ]
    signal_one = signal_one[
        "value", "timestamp_received", "field"
    ]  # select columns we care about
    signal_one = signal_one.rename(current="value", signal_one="field")

    signal_two = table[
        table["field"].contains("signal_two") & table["field"].contains("voltage")
    ]
    signal_two = signal_two[
        "value", "timestamp_received", "field"
    ]  # select columns we care about
    signal_two = signal_two.rename(voltage="value", signal_two="field")

    merged = ibis.api.asof_join(signal_one, signal_two, "timestamp_received")
    assert merged.columns == [
        "current",
        "timestamp_received",
        "signal_one",
        "voltage",
        "timestamp_received_right",
        "signal_two",
    ]


def test_select_from_unambiguous_join_with_strings():
    # GH1387
    t = ibis.table([("a", "int64"), ("b", "string")])
    s = ibis.table([("b", "int64"), ("c", "string")])
    joined = t.left_join(s, [t.b == s.c])
    expr = joined[t, "c"]
    assert expr.columns == ["a", "b", "c"]


def test_filter_applied_to_join():
    # GH2437
    countries = ibis.table([("iso_alpha3", "string")])
    gdp = ibis.table([("country_code", "string"), ("year", "int64")])

    expr = countries.inner_join(
        gdp,
        predicates=[countries["iso_alpha3"] == gdp["country_code"]],
    ).filter(gdp["year"] == 2017)
    assert expr.columns == ["iso_alpha3", "country_code", "year"]


@pytest.mark.parametrize("how", ["inner", "left", "outer", "right"])
def test_join_lname_rname(how):
    left = ibis.table([("id", "int64"), ("first_name", "string")])
    right = ibis.table([("id", "int64"), ("last_name", "string")])
    method = getattr(left, f"{how}_join")

    expr = method(right)
    assert expr.columns == ["id", "first_name", "id_right", "last_name"]

    expr = method(right, rname="right_{name}")
    assert expr.columns == ["id", "first_name", "right_id", "last_name"]

    expr = method(right, lname="left_{name}", rname="")
    assert expr.columns == ["left_id", "first_name", "id", "last_name"]

    expr = method(right, rname="right_{name}", lname="left_{name}")
    assert expr.columns == ["left_id", "first_name", "right_id", "last_name"]


def test_join_lname_rname_still_collide():
    t1 = ibis.table({"id": "int64", "col1": "int64", "col2": "int64"})
    t2 = ibis.table({"id": "int64", "col1": "int64", "col2": "int64"})
    t3 = ibis.table({"id": "int64", "col1": "int64", "col2": "int64"})

    with pytest.raises(com.IntegrityError) as rec:
        t1.left_join(t2, "id").left_join(t3, "id")

    assert "`['col1_right', 'col2_right', 'id_right']`" in str(rec.value)
    assert "`lname='', rname='{name}_right'`" in str(rec.value)


def test_drop():
    t = ibis.table(dict.fromkeys("abcd", "int"))

    assert t.drop() is t

    res = t.drop("a")
    assert res.equals(t.select("b", "c", "d"))

    res = t.drop("a", "b")
    assert res.equals(t.select("c", "d"))

    assert res.equals(t.select("c", "d"))

    assert res.equals(t.drop(s.matches("a|b")))

    res = t.drop(_.a)
    assert res.equals(t.select("b", "c", "d"))

    res = t.drop(_.a, _.b)
    assert res.equals(t.select("c", "d"))

    res = t.drop(_.a, "b")
    assert res.equals(t.select("c", "d"))

    with pytest.raises(KeyError):
        t.drop("e")


def test_python_table_ambiguous():
    with pytest.raises(NotImplementedError):
        ibis.memtable(
            [(1,)],
            schema=ibis.schema(dict(a="int8")),
            columns=["a"],
        )


def test_memtable_filter():
    # Mostly just a smoketest, this used to error on construction
    t = ibis.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.filter(t.x > 1)
    assert expr.columns == ["x", "y"]


def test_default_backend_with_unbound_table():
    t = ibis.table(dict(a="int"), name="t")
    expr = t.a.sum()

    with pytest.raises(
        com.IbisError,
        match="Expression contains unbound tables",
    ):
        assert expr.execute()


def test_numpy_ufuncs_dont_cast_tables():
    t = ibis.table(dict.fromkeys("abcd", "int"))
    for arg in [np.int64(1), np.array([1, 2, 3])]:
        for left, right in [(t, arg), (arg, t)]:
            with pytest.raises(TypeError):
                left + right


def test_array_string_compare():
    t = ibis.table(schema=dict(by="string", words="array<string>"), name="t")
    expr = t[t.by == "foo"].mutate(words=_.words.unnest()).filter(_.words == "the")
    assert expr is not None


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    "api",
    [
        param(lambda t, value: t[value], id="getitem"),
        param(lambda t, value: t.filter(value), id="filter"),
    ],
)
def test_filter_with_literal(value, api):
    t = ibis.table(dict(a="string"))
    filt = api(t, ibis.literal(value))
    assert filt is not None

    # ints are invalid predicates
    int_val = ibis.literal(int(value))
    with pytest.raises((NotImplementedError, ValidationError, com.IbisTypeError)):
        api(t, int_val)


def test_cast():
    t = ibis.table(dict(a="int", b="string", c="float"), name="t")

    assert t.cast({"a": "string"}).equals(t.mutate(a=t.a.cast("string")))

    with pytest.raises(
        com.IbisError, match="fields that are not in the table: .+'d'.+"
    ):
        t.cast({"d": "array<int>"}).equals(t.select())

    assert t.cast(ibis.schema({"a": "string", "b": "int"})).equals(
        t.mutate(a=t.a.cast("string"), b=t.b.cast("int"))
    )
    assert t.cast([("a", "string"), ("b", "float")]).equals(
        t.mutate(a=t.a.cast("string"), b=t.b.cast("float"))
    )


def test_pivot_longer():
    diamonds = ibis.table(
        {
            "carat": "float64",
            "cut": "string",
            "color": "string",
            "clarity": "string",
            "depth": "float64",
            "table": "float64",
            "price": "int64",
            "x": "float64",
            "y": "float64",
            "z": "float64",
        },
        name="diamonds",
    )
    res = diamonds.pivot_longer(s.c("x", "y", "z"), names_to="pos", values_to="xyz")
    assert res.schema().names == (
        "carat",
        "cut",
        "color",
        "clarity",
        "depth",
        "table",
        "price",
        "pos",
        "xyz",
    )


def test_pivot_longer_strip_prefix():
    t = ibis.table(
        dict(artist="string", track="string", wk1="int", wk2="int", wk3="int")
    )
    expr = t.pivot_longer(
        s.startswith("wk"),
        names_to="week",
        names_pattern=r"wk(.+)",
        names_transform=int,
        values_to="rank",
        values_transform=_.cast("int"),
    )
    schema = ibis.schema(dict(artist="string", track="string", week="int8", rank="int"))
    assert expr.schema() == schema


def test_pivot_longer_pluck_regex():
    t = ibis.table(
        dict(artist="string", track="string", x_wk1="int", x_wk2="int", x_wk3="int")
    )
    expr = t.pivot_longer(
        s.matches("^.+wk.$"),
        names_to=["other_var", "week"],
        names_pattern=r"(.)_wk(\d)",
        names_transform=dict(other_var=str.upper, week=int),
        values_to="rank",
        values_transform=_.cast("int"),
    )
    schema = ibis.schema(
        dict(
            artist="string", track="string", other_var="string", week="int8", rank="int"
        )
    )
    assert expr.schema() == schema


def test_pivot_longer_no_match():
    t = ibis.table(
        dict(artist="string", track="string", x_wk1="int", x_wk2="int", x_wk3="int")
    )
    with pytest.raises(
        com.IbisInputError, match="Selector returned no columns to pivot on"
    ):
        t.pivot_longer(
            s.matches("foo"),
            names_to=["other_var", "week"],
            names_pattern=r"(.)_wk(\d)",
            names_transform=dict(other_var=str.upper, week=int),
            values_to="rank",
            values_transform=_.cast("int"),
        )


def test_pivot_wider():
    fish = ibis.table({"fish": "int", "station": "string", "seen": "int"}, name="fish")
    res = fish.pivot_wider(
        names=["Release", "Lisbon"], names_from="station", values_from="seen"
    )
    assert res.schema().names == ("fish", "Release", "Lisbon")
    with pytest.raises(com.IbisInputError, match="Columns .+ are not present in"):
        fish.pivot_wider(names=["Release", "Lisbon"], values_from="seen")


def test_invalid_deferred():
    t = ibis.table(dict(value="int", lagged_value="int"), name="t")

    with pytest.raises(ValidationError):
        ops.Greatest((t.value, ibis._.lagged_value))


@pytest.mark.parametrize("keep", ["last", None])
def test_invalid_distinct(keep):
    t = ibis.table(dict(a="int"), name="t")
    with pytest.raises(com.IbisError, match="Only keep='first'"):
        t.distinct(keep=keep)


def test_invalid_keep_distinct():
    t = ibis.table(dict(a="int", b="string"), name="t")
    with pytest.raises(com.IbisError, match="Invalid value for `keep`:"):
        t.distinct(on="a", keep="invalid")


def test_invalid_distinct_empty_key():
    t = ibis.table(dict(a="int", b="string"), name="t")
    with pytest.raises(com.IbisInputError):
        t.distinct(on="c", keep="first")
