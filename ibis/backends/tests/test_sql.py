from __future__ import annotations

import io

import pytest
from pytest import mark, param

import ibis
import ibis.common.exceptions as exc
from ibis import _
from ibis.backends.conftest import _get_backends_to_test

sa = pytest.importorskip("sqlalchemy")
sg = pytest.importorskip("sqlglot")

pytestmark = pytest.mark.notimpl(["druid"])


@mark.never(
    ["dask", "pandas"],
    reason="Dask and Pandas are not SQL backends",
    raises=(NotImplementedError, ValueError),
)
@mark.notimpl(
    ["datafusion", "pyspark", "polars"],
    reason="Not clear how to extract SQL from the backend",
    raises=(exc.OperationNotDefinedError, NotImplementedError, ValueError),
)
def test_table(backend):
    expr = backend.functional_alltypes.select(c=_.int_col + 1)
    buf = io.StringIO()
    ibis.show_sql(expr, file=buf)
    assert buf.getvalue()


simple_literal = param(ibis.literal(1), id="simple_literal")
array_literal = param(
    ibis.array([1]),
    marks=[
        mark.never(
            ["mysql", "mssql", "oracle"],
            raises=sa.exc.CompileError,
            reason="arrays not supported in the backend",
        ),
        mark.notyet(
            ["impala", "sqlite"],
            raises=NotImplementedError,
            reason="backends hasn't implemented array literals",
        ),
    ],
    id="array_literal",
)
no_structs = mark.never(
    ["impala", "mysql", "sqlite", "mssql"],
    raises=(NotImplementedError, sa.exc.CompileError),
    reason="structs not supported in the backend",
)
no_struct_literals = mark.notimpl(
    ["postgres", "mssql", "oracle"], reason="struct literals are not yet implemented"
)
not_sql = mark.never(
    ["pandas", "dask"],
    raises=(exc.IbisError, NotImplementedError, ValueError),
    reason="Not a SQL backend",
)
no_sql_extraction = mark.notimpl(
    ["datafusion", "pyspark", "polars"],
    reason="Not clear how to extract SQL from the backend",
)


@mark.parametrize(
    "expr",
    [
        simple_literal,
        array_literal,
        param(
            ibis.struct(dict(a=1)),
            marks=[no_structs, no_struct_literals],
            id="struct_literal",
        ),
    ],
)
@not_sql
@no_sql_extraction
def test_literal(backend, expr):
    assert ibis.to_sql(expr, dialect=backend.name())


@pytest.mark.never(
    ["pandas", "dask", "datafusion", "polars", "pyspark"], reason="not SQL"
)
@pytest.mark.xfail_version(
    mssql=["sqlalchemy>=2"], reason="sqlalchemy 2 prefixes literals with `N`"
)
def test_group_by_has_index(backend, snapshot):
    countries = ibis.table(
        dict(continent="string", population="int64"), name="countries"
    )
    expr = countries.group_by(
        cont=(
            _.continent.case()
            .when("NA", "North America")
            .when("SA", "South America")
            .when("EU", "Europe")
            .when("AF", "Africa")
            .when("AS", "Asia")
            .when("OC", "Oceania")
            .when("AN", "Antarctica")
            .else_("Unknown continent")
            .end()
        )
    ).agg(total_pop=_.population.sum())
    sql = str(ibis.to_sql(expr, dialect=backend.name()))
    snapshot.assert_match(sql, "out.sql")


@pytest.mark.never(
    ["pandas", "dask", "datafusion", "polars", "pyspark"], reason="not SQL"
)
def test_cte_refs_in_topo_order(backend, snapshot):
    mr0 = ibis.table(schema=ibis.schema(dict(key="int")), name="leaf")

    mr1 = mr0.filter(ibis.literal(True))

    mr2 = mr1.join(mr1[["key"]], ["key"])
    mr3 = mr2.join(mr2, ["key"])

    sql = str(ibis.to_sql(mr3, dialect=backend.name()))
    snapshot.assert_match(sql, "out.sql")


@pytest.mark.never(
    ["pandas", "dask", "datafusion", "polars", "pyspark"], reason="not SQL"
)
def test_isin_bug(con, snapshot):
    t = ibis.table(dict(x="int"), name="t")
    good = t[t.x > 2].x
    expr = t.x.isin(good)
    snapshot.assert_match(str(ibis.to_sql(expr, dialect=con.name)), "out.sql")


@pytest.mark.never(
    ["pandas", "dask", "datafusion", "polars", "pyspark"],
    reason="not SQL",
    raises=NotImplementedError,
)
@pytest.mark.notyet(
    ["sqlite", "mysql", "druid", "impala", "mssql"], reason="no unnest support upstream"
)
@pytest.mark.notimpl(
    ["oracle"], reason="unnest not yet implemented", raises=exc.OperationNotDefinedError
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
