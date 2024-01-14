from __future__ import annotations

import os
from contextlib import closing
from posixpath import join as pjoin

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import util
from ibis.tests.util import assert_equal

pytest.importorskip("impala")

from impala.error import HiveServer2Error  # noqa: E402


@pytest.fixture
def temp_view(con) -> str:
    name = util.gen_name("view")
    yield name
    con.drop_view(name, force=True)


def test_create_exists_view(con, temp_view):
    assert temp_view not in con.list_tables()

    t1 = con.table("functional_alltypes").limit(1)
    t2 = con.create_view(temp_view, t1)

    assert temp_view in con.list_tables()
    assert not t2.execute().empty


def test_drop_non_empty_database(con, alltypes, temp_table):
    con.create_table(temp_table, alltypes)
    assert temp_table in con.list_tables()

    with pytest.raises(com.IntegrityError):
        con.drop_database("ibis_testing")


@pytest.fixture
def temp_db(con, tmp_dir):
    base = pjoin(tmp_dir, util.guid())
    name = util.gen_name("test_database")
    yield pjoin(base, name)
    con.drop_database(name)


def test_create_database_with_location(con, temp_db):
    name = os.path.basename(temp_db)
    con.create_database(name, path=temp_db)
    assert name in con.list_databases()


def test_create_table_with_location_execute(
    con, tmp_dir, alltypes, test_data_db, temp_table
):
    base = pjoin(tmp_dir, util.guid())
    name = f"test_{util.guid()}"
    tmp_path = pjoin(base, name)

    expr = alltypes

    con.create_table(temp_table, obj=expr, location=tmp_path, database=test_data_db)
    assert temp_table in con.list_tables()


def test_drop_table_not_exist(con):
    non_existent_table = f"ibis_table_{util.guid()}"
    with pytest.raises(HiveServer2Error):
        con.drop_table(non_existent_table)
    con.drop_table(non_existent_table, force=True)


def test_truncate_table(con, alltypes, temp_table):
    expr = alltypes.limit(1)

    con.create_table(temp_table, obj=expr)

    try:
        con.truncate_table(temp_table)
    except HiveServer2Error as e:
        if "AnalysisException" in e.args[0]:
            pytest.skip("TRUNCATE not available in this version of Impala")

    t = con.table(temp_table)
    nrows = t.count().execute()
    assert not nrows


def test_truncate_table_expression(con, alltypes, temp_table):
    expr = alltypes.limit(1)

    con.create_table(temp_table, obj=expr)
    t = con.table(temp_table)
    t.truncate()
    nrows = t.count().execute()
    assert not nrows


def test_ctas_from_table_expr(con, alltypes, temp_table):
    t = con.create_table(temp_table, alltypes)
    assert t.count().execute()


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

    # check using ImpalaTable.insert
    t = con.table(temp_table, database=db)
    t.insert(expr.limit(10))

    sz = t.count()
    assert sz.execute() == 20

    # Overwrite and verify only 10 rows now
    t.insert(expr.limit(10), overwrite=True)
    assert sz.execute() == 10


def test_insert_validate_types(con, alltypes, test_data_db, temp_table):
    # GH #235
    db = test_data_db

    expr = alltypes
    con.create_table(
        temp_table,
        schema=expr["tinyint_col", "int_col", "string_col"].schema(),
        database=db,
    )

    t = con.table(temp_table, database=db)

    to_insert = expr[
        expr.tinyint_col, expr.smallint_col.name("int_col"), expr.string_col
    ]
    t.insert(to_insert.limit(10))

    to_insert = expr[
        expr.tinyint_col,
        expr.smallint_col.cast("int32").name("int_col"),
        expr.string_col,
    ]
    t.insert(to_insert.limit(10))

    to_insert = expr[expr.tinyint_col, expr.bigint_col.name("int_col"), expr.string_col]

    limit_expr = to_insert.limit(10)
    with pytest.raises(com.IbisError):
        t.insert(limit_expr)


def test_compute_stats(con):
    t = con.table("functional_alltypes")

    t.compute_stats()
    t.compute_stats(incremental=True)

    con.compute_stats("functional_alltypes")


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
def path_uuid():
    return f"change-location-{util.guid()}"


@pytest.fixture
def table(con, tmp_dir, path_uuid):
    table_name = f"table_{util.guid()}"
    fake_path = pjoin(tmp_dir, path_uuid)
    schema = ibis.schema([("foo", "string"), ("bar", "int64")])
    yield con.create_table(
        table_name, schema=schema, format="parquet", external=True, location=fake_path
    )
    con.drop_table(table_name)


def test_change_location(table, tmp_dir, path_uuid):
    old_loc = table.metadata().location

    new_path = pjoin(tmp_dir, "new-path")
    table.alter(location=new_path)

    new_loc = table.metadata().location
    assert new_loc == old_loc.replace(path_uuid, "new-path")


def test_change_properties(table):
    props = {"foo": "1", "bar": "2"}

    table.alter(tbl_properties=props)
    tbl_props = table.metadata().tbl_properties
    for k, v in props.items():
        assert v == tbl_props[k]

    table.alter(serde_properties=props)
    serde_props = table.metadata().serde_properties
    for k, v in props.items():
        assert v == serde_props[k]


def test_change_format(table):
    table.alter(format="avro")

    meta = table.metadata()
    assert "Avro" in meta.hive_format


def test_query_avro(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, "impala/avro/tpch/region")

    avro_schema = {
        "fields": [
            {"type": ["int", "null"], "name": "R_REGIONKEY"},
            {"type": ["string", "null"], "name": "R_NAME"},
            {"type": ["string", "null"], "name": "R_COMMENT"},
        ],
        "type": "record",
        "name": "a",
    }

    table = con.avro_file(hdfs_path, avro_schema)

    # table exists
    assert table._qualified_name in con.list_tables()

    expr = table.r_name.value_counts()
    expr.execute()

    assert table.count().execute() == 5

    df = table.execute()
    assert len(df) == 5


@pytest.fixture
def temp_table_id(con):
    name = "distinct"
    yield name
    con.drop_table(name)


def test_create_table_reserved_identifier(con, temp_table_id):
    expr = con.table("functional_alltypes")
    expected = expr.count().execute()
    con.create_table(temp_table_id, expr)
    result = con.table(temp_table_id).count().execute()
    assert result == expected


def test_query_delimited_file_directory(con, test_data_dir, temp_table):
    hdfs_path = pjoin(test_data_dir, "csv")

    schema = ibis.schema([("foo", "string"), ("bar", "double"), ("baz", "int8")])
    table = con.delimited_file(hdfs_path, schema, name=temp_table, delimiter=",")

    expr = (
        table[table.bar > 0]
        .group_by("foo")
        .aggregate(
            [
                table.bar.sum().name("sum(bar)"),
                table.baz.sum().name("mean(baz)"),
            ]
        )
    )
    assert expr.execute() is not None


@pytest.fixture
def temp_char_table(con):
    name = "testing_varchar_support"
    with closing(
        con.raw_sql(
            f"""\
CREATE TABLE IF NOT EXISTS {name} (
  group1 VARCHAR(10),
  group2 CHAR(10)
)"""
        )
    ):
        pass
    assert name in con.list_tables(), name
    yield con.table(name)
    con.drop_table(name, force=True)


def test_varchar_char_support(temp_char_table):
    assert isinstance(temp_char_table["group1"], ir.StringValue)
    assert isinstance(temp_char_table["group2"], ir.StringValue)


def test_access_kudu_table(kudu_table):
    assert kudu_table.columns == ["a"]
    assert kudu_table["a"].type() == dt.string
