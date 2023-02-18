import io

import pytest
from pytest import mark, param

import ibis
import ibis.common.exceptions as exc
from ibis import _

sa = pytest.importorskip("sqlalchemy")
pytest.importorskip("sqlglot")


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
@mark.notimpl(["mssql"], raises=ValueError, reason="no sqlglot dialect for mssql")
def test_table(con):
    expr = con.tables.functional_alltypes.select(c=_.int_col + 1)
    buf = io.StringIO()
    ibis.show_sql(expr, file=buf)
    assert buf.getvalue()


simple_literal = param(
    ibis.literal(1),
    marks=[pytest.mark.notimpl(["mssql"], reason="no sqlglot dialect for mssql")],
    id="simple_literal",
)
array_literal = param(
    ibis.array([1]),
    marks=[
        mark.never(
            ["mysql", "mssql"],
            raises=sa.exc.CompileError,
            reason="arrays not supported in the backend",
        ),
        mark.notyet(
            ["impala", "sqlite"],
            raises=NotImplementedError,
            reason="backends hasn't implemented array literals",
        ),
        mark.notimpl(
            ["trino"], reason="Cannot render array literals", raises=sa.exc.CompileError
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
    ["postgres", "mssql"], reason="struct literals are not yet implemented"
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
@pytest.mark.notyet(["mssql"], reason="sqlglot doesn't support an mssql dialect")
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
@pytest.mark.notimpl(
    ["mssql"], reason="sqlglot dialect not yet implemented", raises=ValueError
)
def test_cte_refs_in_topo_order(backend, snapshot):
    mr0 = ibis.table(schema=ibis.schema(dict(key="int")), name='leaf')

    mr1 = mr0.filter(ibis.literal(True))

    mr2 = mr1.join(mr1[['key']], ["key"])
    mr3 = mr2.join(mr2, ['key'])

    sql = str(ibis.to_sql(mr3, dialect=backend.name()))
    snapshot.assert_match(sql, "out.sql")
