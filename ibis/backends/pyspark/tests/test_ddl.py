from __future__ import annotations

import os
import shutil
from posixpath import join as pjoin

import pytest

import ibis
from ibis import util
from ibis.tests.util import assert_equal

pyspark = pytest.importorskip("pyspark")


@pytest.fixture
def temp_view(con) -> str:
    name = util.gen_name("view")
    yield name
    con.drop_view(name, force=True)


def test_create_exists_view(con, alltypes, temp_view):
    assert temp_view not in con.list_tables()

    t1 = alltypes.group_by("string_col").size()
    t2 = con.create_view(temp_view, t1)

    assert temp_view in con.list_tables()
    # just check it works for now
    assert t2.execute() is not None


def test_drop_non_empty_database(con, alltypes, temp_table_db):
    temp_database, temp_table = temp_table_db
    con.create_table(temp_table, alltypes, database=temp_database)
    assert temp_table in con.list_tables(database=temp_database)

    with pytest.raises(pyspark.sql.utils.AnalysisException):
        con.drop_database(temp_database)


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
    with pytest.raises(pyspark.sql.utils.AnalysisException):
        con.drop_table(non_existent_table)
    con.drop_table(non_existent_table, force=True)


def test_truncate_table(con, alltypes, temp_table):
    expr = alltypes.limit(1)

    con.create_table(temp_table, obj=expr)
    con.truncate_table(temp_table)

    t = con.table(temp_table)
    nrows = t.count().execute()
    assert not nrows


def test_ctas_from_table_expr(con, alltypes, temp_table_db):
    expr = alltypes
    db, table_name = temp_table_db

    con.create_table(table_name, expr, database=db)


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


def test_insert_table(con, alltypes, temp_table, test_data_db):
    expr = alltypes
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


def test_compute_stats(con, alltypes, temp_table):
    con.create_table(temp_table, alltypes)
    con.compute_stats(temp_table)
    con.compute_stats(temp_table, noscan=True)


@pytest.fixture
def created_view(con, alltypes):
    name = util.guid()
    expr = alltypes.limit(10)
    con.create_view(name, expr)
    return name


def test_drop_view(con, created_view):
    con.drop_view(created_view)
    assert created_view not in con.list_tables()


@pytest.fixture
def table(con, temp_database):
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


def test_create_table_reserved_identifier(con, alltypes, keyword_t):
    expr = alltypes
    expected = expr.count().execute()
    t = con.create_table(keyword_t, expr)
    result = t.count().execute()
    assert result == expected
