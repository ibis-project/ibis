from __future__ import annotations

import pandas as pd
import pytest
from pytest import param

import ibis
from ibis import literal as L
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("alltypes")


@pytest.mark.parametrize(
    "value",
    [
        param("simple", id="simple"),
        param("I can't", id="embedded_single_quote"),
        param('An "escape"', id="embedded_double_quote"),
        param(5, id="int"),
        param(1.5, id="float"),
        param(True, id="true"),
        param(False, id="false"),
    ],
)
def test_literals(value, snapshot):
    expr = L(value)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_column_ref_table_aliases(snapshot):
    context = ImpalaCompiler.make_context()

    table1 = ibis.table([("key1", "string"), ("value1", "double")])

    table2 = ibis.table([("key2", "string"), ("value and2", "double")])

    context.set_ref(table1.op(), "t0")
    context.set_ref(table2.op(), "t1")

    expr = table1["value1"] - table2["value and2"]

    result = translate(expr, context=context)
    snapshot.assert_match(result, "out.sql")


def test_column_ref_quoting():
    schema = [("has a space", "double")]
    table = ibis.table(schema)
    translate(table["has a space"], named="`has a space`")


def test_identifier_quoting():
    schema = [("date", "double"), ("table", "string")]
    table = ibis.table(schema)
    translate(table["date"], named="`date`")
    translate(table["table"], named="`table`")


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda t: t.g.cast("double").name("g_dub"),
        lambda t: t.g.name("has a space"),
        lambda t: ((t.a - t.b) * t.a).name("expr"),
    ],
    ids=["cast", "spaces", "compound_expr"],
)
def test_named_expressions(table, expr_fn, snapshot):
    expr = expr_fn(table)
    result = translate(expr, named=True)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda t: t.a + t.b, id="add"),
        param(lambda t: t.a - t.b, id="sub"),
        param(lambda t: t.a * t.b, id="mul"),
        param(lambda t: t.a / t.b, id="div"),
        param(lambda t: t.a**t.b, id="pow"),
        param(lambda t: t.a < t.b, id="lt"),
        param(lambda t: t.a <= t.b, id="le"),
        param(lambda t: t.a > t.b, id="gt"),
        param(lambda t: t.a >= t.b, id="ge"),
        param(lambda t: t.a == t.b, id="eq"),
        param(lambda t: t.a != t.b, id="ne"),
        param(lambda t: t.h & (t.a > 0), id="and"),
        param(lambda t: t.h | (t.a > 0), id="or"),
        param(lambda t: t.h ^ (t.a > 0), id="xor"),
    ],
)
def test_binary_infix_operators(table, expr_fn, snapshot):
    expr = expr_fn(table)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda t: (t.a + t.b) + t.c,
        lambda t: t.a.log() + t.c,
        lambda t: t.b + (-(t.a + t.c)),
    ],
    ids=["parens_left", "function_call", "negation"],
)
def test_binary_infix_parenthesization(table, expr_fn, snapshot):
    expr = expr_fn(table)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_between(table, snapshot):
    expr = table.f.between(0, 1)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda t: t["g"].isnull(),
        lambda t: t["a"].notnull(),
        lambda t: (t["a"] + t["b"]).isnull(),
    ],
    ids=["isnull", "notnull", "compound_isnull"],
)
def test_isnull_notnull(table, expr_fn, snapshot):
    expr = expr_fn(table)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    ("column", "to_type"),
    [
        ("a", "int16"),
        ("a", "int32"),
        ("a", "int64"),
        ("a", "string"),
        ("d", "int8"),
        ("g", "double"),
        ("g", "timestamp"),
    ],
)
def test_casts(table, column, to_type, snapshot):
    expr = table[column].cast(to_type)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_misc_conditionals(table, snapshot):
    expr = table.a.nullif(0)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda _: L("9.9999999").cast("decimal(38, 5)"),
        lambda t: t.f.cast("decimal(12, 2)"),
    ],
    ids=["literal", "column"],
)
def test_decimal_casts(table, expr_fn, snapshot):
    expr = expr_fn(table)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("colname", ["a", "f", "h"])
def test_negate(table, colname, snapshot):
    result = translate(-table[colname])
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "field",
    ["year", "month", "day", "hour", "minute", "second", "microsecond", "millisecond"],
)
def test_timestamp_extract_field(table, field, snapshot):
    expr = getattr(table.i, field)()
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_sql_extract(table, snapshot):
    # integration with SQL translation
    expr = table[
        table.i.year().name("year"),
        table.i.month().name("month"),
        table.i.day().name("day"),
    ]

    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_timestamp_now(snapshot):
    expr = ibis.now()
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "unit",
    ["years", "months", "weeks", "days", "hours", "minutes", "seconds"],
)
def test_timestamp_deltas(table, unit, snapshot):
    K = 5

    offset = ibis.interval(**{unit: K})

    add_expr = table.i + offset
    result = translate(add_expr)
    snapshot.assert_match(result, "out1.sql")

    sub_expr = table.i - offset
    result = translate(sub_expr)
    snapshot.assert_match(result, "out2.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda v: L(pd.Timestamp(v)),
        lambda v: L(pd.Timestamp(v).to_pydatetime()),
        lambda v: ibis.timestamp(v),
    ],
    ids=["pd_timestamp", "pydatetime", "timestamp_function"],
)
def test_timestamp_literals(expr_fn, snapshot):
    expr = expr_fn("2015-01-01 12:34:56")
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("method_name", ["index", "full_name"])
def test_timestamp_day_of_week(method_name, snapshot):
    ts = ibis.timestamp("2015-09-01T01:00:23")
    expr = getattr(ts.day_of_week, method_name)()
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda col: col.to_timestamp(),
        lambda col: col.to_timestamp("ms"),
        lambda col: col.to_timestamp("us"),
    ],
    ids=["default", "ms", "us"],
)
def test_timestamp_from_integer(table, expr_fn, snapshot):
    expr = expr_fn(table.c)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_correlated_predicate_subquery(table, snapshot):
    t0 = table
    t1 = t0.view()

    expr = t0.g == t1.g

    ctx = ImpalaCompiler.make_context()
    ctx.make_alias(t0.op())

    # Grab alias from parent context
    subctx = ctx.subcontext()
    subctx.make_alias(t1.op())
    subctx.make_alias(t0.op())

    result = translate(expr, context=subctx)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda b: b.any(), id="any"),
        param(lambda b: -b.any(), id="not_any"),
        param(lambda b: b.all(), id="all"),
        param(lambda b: -b.all(), id="not_all"),
    ],
)
def test_any_all(table, expr_fn, snapshot):
    expr = expr_fn(table.f == 0)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")
