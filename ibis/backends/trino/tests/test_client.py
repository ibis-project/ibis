from __future__ import annotations

import math
import string

import pytest

import ibis
import ibis.common.exceptions as exc
from ibis import udf, util
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
        c.execute(f"SHOW CREATE TABLE {tmp_name}")
        [(ddl,)] = c.fetchall()
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
    assert {"hive", "memory", "system", "tpch", "tpcds"}.issubset(con.list_databases())


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
    assert con.con.source == expected


@pytest.mark.parametrize(
    ("database", "schema", "table"),
    [
        # tables known to exist
        ("system", "metadata", "table_comments"),
        ("tpcds", "sf1", "store"),
        ("tpch", "sf1", "nation"),
    ],
)
def test_cross_schema_table_access(con, database, schema, table):
    t = con.table(table, schema=schema, database=database)
    assert t.count().execute()


def test_builtin_scalar_udf(con, snapshot):
    @udf.scalar.builtin
    def bar(x: float, width: int) -> str:
        """Render a single bar of length `width`, with `x` percent filled."""

    expr = bar(0.25, 40)
    result = con.execute(expr)
    snapshot.assert_match(result, "result.txt")


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def geometric_mean(x) -> float:
        """Geometric mean of a series of numbers."""

    t = con.table("diamonds")
    expr = t.agg(n=t.count(), geomean=geometric_mean(t.price))
    result_n, result = expr.execute().squeeze().tolist()

    with con.begin() as c:
        c.execute("SELECT COUNT(*), GEOMETRIC_MEAN(price) FROM diamonds")
        [(expected_n, expected)] = c.fetchall()

    # check the count
    assert result_n > 0
    assert expected_n > 0
    assert result_n == expected_n

    # check the value
    assert result is not None
    assert expected is not None
    assert math.isfinite(result)
    assert result == expected


def test_create_table_timestamp():
    con = ibis.trino.connect(database="memory", schema="default")
    schema = ibis.schema(
        dict(zip(string.ascii_letters, map("timestamp({:d})".format, range(10))))
    )
    table = util.gen_name("trino_temp_table")
    t = con.create_table(table, schema=schema)
    try:
        rows = con.raw_sql(f"DESCRIBE {table}").fetchall()
        result = ibis.schema((name, typ) for name, typ, *_ in rows)
        assert result == schema
        assert result == t.schema()
    finally:
        con.drop_table(table)
        assert table not in con.list_tables()


def test_table_access_database_schema(con):
    t = con.table("region", schema="sf1", database="tpch")
    assert t.count().execute()

    with pytest.raises(exc.IbisError, match='Table not found: tpch."tpch.sf1".region'):
        con.table("region", schema="tpch.sf1", database="tpch")

    with pytest.raises(
        exc.IbisError, match='Table not found: system."tpch.sf1".region'
    ):
        con.table("region", schema="tpch.sf1", database="system")
