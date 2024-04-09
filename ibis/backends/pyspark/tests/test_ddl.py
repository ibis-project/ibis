from __future__ import annotations

import os
import shutil
from posixpath import join as pjoin

import pytest

import ibis
from ibis import util
from ibis.backends.tests.errors import PySparkAnalysisException
from ibis.tests.util import assert_equal

pyspark = pytest.importorskip("pyspark")


@pytest.fixture
def temp_view(con) -> str:
    name = util.gen_name("view")
    yield name
    con.drop_view(name, force=True)


@pytest.mark.parametrize(
    "table_name",
    [
        pytest.param("functional_alltypes", id="batch"),
        pytest.param(
            "functional_alltypes_streaming",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Streaming aggregations require watermark.",
            ),
            id="streaming",
        ),
    ],
)
def test_create_exists_view(con, temp_view, table_name):
    assert temp_view not in con.list_tables()

    table = con.table(table_name)
    t1 = table.group_by("string_col").size()
    t2 = con.create_view(temp_view, t1)

    assert temp_view in con.list_tables()
    # just check it works for now
    assert t2.execute() is not None


def test_drop_non_empty_database(con, alltypes, temp_table_db):
    temp_database, temp_table = temp_table_db

    con.create_table(temp_table, alltypes, database=temp_database)
    assert temp_table in con.list_tables(database=temp_database)

    with pytest.raises(PySparkAnalysisException):
        con.drop_database(temp_database)


def test_drop_non_empty_database_for_streaming(
    con_streaming, alltypes_streaming, temp_database
):
    db_name = util.gen_name("database")
    con_streaming.create_database(db_name)

    table_name = "alltypes_streaming"
    con_streaming.create_table(table_name, alltypes_streaming, database=temp_database)
    assert table_name in con_streaming.list_tables(database=temp_database)

    # As pyspark backend creates a view for in-memory obj,
    # the database does not contain any actual table, and
    # can be dropped.
    con_streaming.drop_database(db_name)


@pytest.fixture
def temp_base():
    base = pjoin(f"/tmp/{util.gen_name('pyspark_testing')}", util.gen_name("temp_base"))
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def temp_db(con, temp_base):
    name = util.gen_name("temp_db")
    yield pjoin(temp_base, name)
    con.drop_database(name, force=True)


def test_create_database_with_location(con, temp_db):
    base = os.path.dirname(temp_db)
    name = os.path.basename(temp_db)
    con.create_database(name, path=temp_db)
    assert os.path.exists(base)


def test_drop_table_not_exist(con):
    non_existent_table = f"ibis_table_{util.guid()}"
    with pytest.raises(PySparkAnalysisException):
        con.drop_table(non_existent_table)
    con.drop_table(non_existent_table, force=True)


@pytest.mark.parametrize(
    "table_name",
    [
        pytest.param("functional_alltypes", id="batch"),
        pytest.param(
            "functional_alltypes_streaming",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Temp view cannot be truncated.",
            ),
            id="streaming",
        ),
    ],
)
def test_truncate_table(con, temp_table, table_name):
    table = con.table(table_name)
    expr = table.limit(1)

    con.create_table(temp_table, obj=expr)
    con.truncate_table(temp_table)

    t = con.table(temp_table)
    nrows = t.count().execute()
    assert not nrows


@pytest.mark.parametrize(
    "table_name", ["functional_alltypes", "functional_alltypes_streaming"]
)
def test_ctas_from_table_expr(con, temp_table_db, table_name):
    table = con.table(table_name)
    db, table_name = temp_table_db
    con.create_table(table_name, table, database=db)


def test_create_empty_table(con, temp_table):
    schema = ibis.schema(
        [
            ("a", "string"),
            ("b", "timestamp"),
            ("c", "decimal(12, 8)"),
            ("d", "double"),
        ]
    )

    con.create_table(temp_table, schema=schema)

    result_schema = con.get_schema(temp_table)
    assert_equal(result_schema, schema)

    assert con.table(temp_table).execute().empty


@pytest.mark.parametrize(
    "table_name",
    [
        pytest.param("functional_alltypes", id="batch"),
        pytest.param(
            "functional_alltypes_streaming",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Cannot insert into temp view.",
            ),
            id="streaming",
        ),
    ],
)
def test_insert_table(con, temp_table, test_data_db, table_name):
    expr = con.table(table_name)
    db = test_data_db

    con.create_table(temp_table, expr.limit(0), database=db)
    con.insert(temp_table, expr.limit(10), database=db)
    assert con.table(temp_table).count().execute() == 10

    # Overwrite and verify only 10 rows now
    con.insert(temp_table, expr.limit(10), overwrite=True)
    assert con.table(temp_table).count().execute() == 10

    con.insert(temp_table, expr.limit(10), database=db, overwrite=False)
    assert con.table(temp_table).count().execute() == 20


def test_insert_validate_types(con, alltypes, test_data_db, temp_table):
    db = test_data_db

    expr = alltypes
    con.create_table(
        temp_table,
        schema=expr["tinyint_col", "int_col", "string_col"].schema(),
        database=db,
    )

    to_insert = expr[
        expr.tinyint_col, expr.smallint_col.name("int_col"), expr.string_col
    ]
    con.insert(temp_table, to_insert.limit(10))

    to_insert = expr[
        expr.tinyint_col,
        expr.smallint_col.cast("int32").name("int_col"),
        expr.string_col,
    ]
    con.insert(temp_table, to_insert.limit(10))


@pytest.mark.parametrize(
    "table_name",
    [
        pytest.param("functional_alltypes", id="batch"),
        pytest.param(
            "functional_alltypes_streaming",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="'ANALYZE TABLE' cannot be executed on temporary view.",
            ),
            id="streaming",
        ),
    ],
)
def test_compute_stats(con, temp_table, table_name):
    table = con.table(table_name)
    con.create_table(temp_table, table)
    con.compute_stats(temp_table)
    con.compute_stats(temp_table, noscan=True)


@pytest.fixture(params=["functional_alltypes", "functional_alltypes_streaming"])
def created_view(con, request):
    name = util.guid()
    table = con.table(request.param)
    expr = table.limit(10)
    con.create_view(name, expr)
    return name


def test_drop_view(con, created_view):
    con.drop_view(created_view)
    assert created_view not in con.list_tables()


@pytest.fixture
def table(con, temp_database):
    # TODO (mehmet): This fixture does not seem to be used anywhere?

    table_name = f"table_{util.guid()}"
    schema = ibis.schema([("foo", "string"), ("bar", "int64")])
    yield con.create_table(
        table_name, database=temp_database, schema=schema, format="parquet"
    )
    con.drop_table(table_name, database=temp_database)


@pytest.fixture
def keyword_t(con):
    yield "distinct"
    con.drop_table("distinct")


@pytest.mark.parametrize(
    "table_name",
    [
        pytest.param("functional_alltypes", id="batch"),
        pytest.param(
            "functional_alltypes_streaming",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Streaming aggregations require watermark.",
            ),
            id="streaming",
        ),
    ],
)
def test_create_table_reserved_identifier(con, keyword_t, table_name):
    expr = con.table(table_name)
    expected = expr.count().execute()
    t = con.create_table(keyword_t, expr)
    result = t.count().execute()
    assert result == expected
