from __future__ import annotations

import re

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as exc
from ibis import _
from ibis.backends.conftest import _get_backends_to_test

sg = pytest.importorskip("sqlglot")


@pytest.mark.parametrize(
    "expr",
    [
        param(ibis.literal(432), id="simple_literal"),
        param(
            ibis.array([432]),
            marks=[
                pytest.mark.never(
                    ["mysql", "mssql", "oracle", "impala", "sqlite"],
                    raises=(exc.OperationNotDefinedError, exc.UnsupportedBackendType),
                    reason="arrays not supported in the backend",
                ),
            ],
            id="array_literal",
        ),
        param(
            ibis.struct(dict(abc=432)),
            marks=[
                pytest.mark.never(
                    ["impala", "mysql", "sqlite", "mssql", "exasol"],
                    raises=(NotImplementedError, exc.UnsupportedBackendType),
                    reason="structs not supported in the backend",
                ),
                pytest.mark.notimpl(
                    ["mssql"], reason="struct literals are not yet implemented"
                ),
            ],
            id="struct_literal",
        ),
    ],
)
@pytest.mark.never(["polars"], reason="not SQL", raises=ValueError)
def test_literal(backend, expr):
    assert "432" in ibis.to_sql(expr, dialect=backend.name())


@pytest.mark.never(["polars"], reason="not SQL", raises=ValueError)
def test_group_by_has_index(backend, snapshot):
    countries = ibis.table(
        dict(continent="string", population="int64"), name="countries"
    )
    expr = countries.group_by(
        cont=(
            _.continent.cases(
                ("NA", "North America"),
                ("SA", "South America"),
                ("EU", "Europe"),
                ("AF", "Africa"),
                ("AS", "Asia"),
                ("OC", "Oceania"),
                ("AN", "Antarctica"),
                else_="Unknown continent",
            )
        )
    ).agg(total_pop=_.population.sum())
    sql = str(ibis.to_sql(expr, dialect=backend.name()))
    snapshot.assert_match(sql, "out.sql")


@pytest.mark.never(["polars"], reason="not SQL", raises=ValueError)
def test_cte_refs_in_topo_order(backend, snapshot):
    mr0 = ibis.table(schema=ibis.schema(dict(key="int")), name="leaf")

    mr1 = mr0.filter(ibis.literal(True))

    mr2 = mr1.join(mr1[["key"]], ["key"])
    mr3 = mr2.join(mr2, ["key"])

    sql = str(ibis.to_sql(mr3, dialect=backend.name()))
    snapshot.assert_match(sql, "out.sql")


@pytest.mark.never(["polars"], reason="not SQL", raises=ValueError)
def test_isin_bug(con, snapshot):
    t = ibis.table(dict(x="int"), name="t")
    good = t.filter(t.x > 2).x
    expr = t.x.isin(good)
    snapshot.assert_match(str(ibis.to_sql(expr, dialect=con.name)), "out.sql")


@pytest.mark.never(["polars"], reason="not SQL", raises=ValueError)
@pytest.mark.notyet(
    ["exasol", "oracle", "flink"],
    reason="no unnest support",
    raises=exc.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["risingwave"], reason="no arbitrary support", raises=exc.OperationNotDefinedError
)
@pytest.mark.notyet(
    ["sqlite", "mysql", "druid", "impala", "mssql"], reason="no unnest support upstream"
)
@pytest.mark.parametrize("backend_name", _get_backends_to_test())
def test_union_aliasing(backend_name, snapshot):
    if backend_name == "snowflake":
        pytest.skip(
            "pivot requires random names to make snowflake work properly and cannot be snapshotted"
        )
    from ibis import selectors as s

    t = ibis.table(
        {
            "field_of_study": "string",
            "1970-71": "int64",
            "1975-76": "int64",
            "1980-81": "int64",
            "1985-86": "int64",
            "1990-91": "int64",
            "1995-96": "int64",
            "2000-01": "int64",
            "2005-06": "int64",
            "2010-11": "int64",
            "2011-12": "int64",
            "2012-13": "int64",
            "2013-14": "int64",
            "2014-15": "int64",
            "2015-16": "int64",
            "2016-17": "int64",
            "2017-18": "int64",
            "2018-19": "int64",
            "2019-20": "int64",
        },
        name="humanities",
    )

    years_window = ibis.window(order_by=_.years, group_by=_.field_of_study)
    diff_agg = (
        t.pivot_longer(s.matches(r"\d{4}-\d{2}"), names_to="years", values_to="degrees")
        .mutate(
            earliest_degrees=_.degrees.first().over(years_window),
            latest_degrees=_.degrees.last().over(years_window),
        )
        .mutate(diff=_.latest_degrees - _.earliest_degrees)
        .group_by(_.field_of_study)
        .agg(diff=_.diff.arbitrary())
    )

    top_ten = diff_agg.order_by(_.diff.desc()).limit(10)
    bottom_ten = diff_agg.filter(_.diff < 0).order_by(_.diff).limit(10)

    result = top_ten.union(bottom_ten)

    snapshot.assert_match(str(ibis.to_sql(result, dialect=backend_name)), "out.sql")


@pytest.mark.never(["polars"], reason="not SQL", raises=ValueError)
@pytest.mark.parametrize(
    "value",
    [
        param(
            ibis.random(),
            marks=pytest.mark.notimpl(
                ["risingwave", "druid"], raises=exc.OperationNotDefinedError
            ),
            id="random",
        ),
        param(
            ibis.uuid(),
            marks=pytest.mark.notimpl(
                ["exasol", "risingwave", "druid", "oracle", "pyspark"],
                raises=exc.OperationNotDefinedError,
            ),
            id="uuid",
        ),
    ],
)
def test_selects_with_impure_operations_not_merged(con, snapshot, value):
    t = ibis.table({"x": "int64", "y": "float64"}, name="t")
    t = t.mutate(y=value, z=value)
    t = t.mutate(size=(t.y == t.z).ifelse("big", "small"))

    sql = str(ibis.to_sql(t, dialect=con.name))
    snapshot.assert_match(sql, "out.sql")


@pytest.mark.never(["polars"], reason="not SQL", raises=NotImplementedError)
def test_to_sql_default_backend(con, snapshot, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    t = ibis.memtable({"b": [1, 2]}, name="mytable")
    expr = t.select("b").count()
    snapshot.assert_match(ibis.to_sql(expr), "to_sql.sql")


@pytest.mark.notimpl(["polars"], raises=ValueError, reason="not a SQL backend")
def test_many_subqueries(backend_name, snapshot):
    def query(t, group_cols):
        t2 = t.mutate(key=ibis.row_number().over(ibis.window(order_by=group_cols)))
        return t2.inner_join(t2[["key"]], "key")

    t = ibis.table(dict(street="str"), name="data")

    t2 = query(t, group_cols=["street"])
    t3 = query(t2, group_cols=["street"])

    snapshot.assert_match(str(ibis.to_sql(t3, dialect=backend_name)), "out.sql")


@pytest.mark.parametrize("backend_name", _get_backends_to_test())
@pytest.mark.notimpl(["polars"], raises=ValueError, reason="not a SQL backend")
def test_mixed_qualified_and_unqualified_predicates(backend_name, snapshot):
    t = ibis.table({"x": "int64"}, name="t")
    expr = t.mutate(y=t.x.sum().over(ibis.window())).filter(
        _.y <= 37, _.x.mean().over().notnull()
    )
    result = ibis.to_sql(expr, dialect=backend_name)

    sc = ibis.backends.sql.compilers
    compiler = getattr(sc, backend_name).compiler

    assert (not compiler.supports_qualify) or re.search(
        r"\bQUALIFY\b", result, flags=re.MULTILINE | re.IGNORECASE
    )
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("backend_name", _get_backends_to_test())
@pytest.mark.notimpl(["polars"], raises=ValueError, reason="not a SQL backend")
@pytest.mark.notimpl(
    ["druid", "risingwave"],
    raises=exc.OperationNotDefinedError,
    reason="random not supported",
)
def test_rewrite_context(snapshot, backend_name):
    table = ibis.memtable({"test": [1, 2, 3, 4, 5]}, name="test")
    expr = table.select(new_col=ibis.ntile(2).over(order_by=ibis.random())).limit(10)
    result = ibis.to_sql(expr, dialect=backend_name)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("subquery", [False, True], ids=["subquery", "table"])
@pytest.mark.parametrize("backend_name", _get_backends_to_test())
@pytest.mark.notimpl(["polars"], raises=ValueError, reason="not a SQL backend")
@pytest.mark.notimpl(
    ["druid", "risingwave"],
    raises=exc.OperationNotDefinedError,
    reason="sample not supported",
)
def test_sample(backend_name, snapshot, subquery):
    t = ibis.table({"x": "int64", "y": "int64"}, name="test")
    if subquery:
        t = t.filter(t.x > 10)
    block = ibis.to_sql(t.sample(0.5, method="block"), dialect=backend_name)
    row = ibis.to_sql(t.sample(0.5, method="row"), dialect=backend_name)
    snapshot.assert_match(block, "block.sql")
    snapshot.assert_match(row, "row.sql")


@pytest.mark.parametrize("backend_name", _get_backends_to_test())
@pytest.mark.notimpl(["polars"], raises=ValueError, reason="not a SQL backend")
def test_order_by_no_deference_literals(backend_name, snapshot):
    t = ibis.table({"a": "int"}, name="test")
    s = t.select("a", i=ibis.literal(9), s=ibis.literal("foo"))
    o = s.order_by("a", "i", "s")
    sql = ibis.to_sql(o, dialect=backend_name)
    snapshot.assert_match(sql, "out.sql")
