from __future__ import annotations

import pytest

import ibis
from ibis.backends.trino.tests.conftest import (
    TRINO_HOST,
    TRINO_PASS,
    TRINO_PORT,
    TRINO_USER,
)


@pytest.fixture
def tmp_name(con):
    name = ibis.util.gen_name("test_trino")
    yield name
    con.drop_table(name, force=True)


def test_table_properties(tmp_name):
    con = ibis.trino.connect(database="hive", schema="default")
    schema = ibis.schema(dict(a="int"))
    t = con.create_table(
        tmp_name,
        schema=schema,
        properties={
            "format": "ORC",
            "bucketed_by": ["a"],
            "bucket_count": 42,
            "extra_properties": {
                "any": "property",
                "you": "want",
            },
        },
    )
    assert t.schema() == schema
    with con.begin() as c:
        ddl = c.exec_driver_sql(f"SHOW CREATE TABLE {tmp_name}").scalar()
    assert "ORC" in ddl
    assert "bucketed_by" in ddl


def test_hive_table_overwrite(tmp_name):
    con = ibis.trino.connect(database="hive", schema="default")
    schema = ibis.schema(dict(a="int"))

    t = con.create_table(tmp_name, schema=schema)
    assert tmp_name in con.list_tables()
    assert t.schema() == schema

    t = con.create_table(tmp_name, schema=schema, overwrite=True)
    assert tmp_name in con.list_tables()
    assert t.schema() == schema


def test_list_catalogs(con):
    assert {"hive", "postgresql", "memory", "system", "tpch", "tpcds"}.issubset(
        con.list_databases()
    )


def test_list_schemas(con):
    assert {"information_schema", "sf1"}.issubset(con.list_schemas(database="tpch"))


@pytest.mark.parametrize(("source", "expected"), [(None, "ibis"), ("foo", "foo")])
def test_con_source(source, expected):
    con = ibis.trino.connect(
        user=TRINO_USER,
        host=TRINO_HOST,
        port=TRINO_PORT,
        password=TRINO_PASS,
        database="hive",
        schema="default",
        source=source,
    )
    assert con.con.url.query["source"] == expected


@pytest.mark.parametrize(
    ("schema", "table"),
    [
        # tables known to exist
        ("memory.default", "diamonds"),
        ("postgresql.public", "map"),
        ("system.metadata", "table_comments"),
        ("tpcds.sf1", "store"),
        ("tpch.sf1", "nation"),
    ],
)
def test_cross_schema_table_access(con, schema, table):
    t = con.table(table, schema=schema)
    assert t.count().execute()
