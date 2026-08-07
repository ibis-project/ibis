from __future__ import annotations

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.starrocks.tests.conftest import (
    IBIS_TEST_STARROCKS_DB,
    STARROCKS_HOST,
    STARROCKS_PASS,
    STARROCKS_PORT,
    STARROCKS_USER,
)


def test_get_schema_from_query(con):
    raw_name = ibis.util.guid()
    con.raw_sql(
        f"""
        CREATE TABLE `{raw_name}` (
          x LARGEINT
        )
        ENGINE=OLAP
        DISTRIBUTED BY RANDOM
        PROPERTIES ("replication_num" = "1")
        """
    ).close()

    try:
        result = con._get_schema_using_query(f"SELECT * FROM `{raw_name}`")

        assert result == ibis.schema({"x": dt.Decimal(38, 0)})
    finally:
        con.drop_table(raw_name, force=True)


def test_create_table_includes_starrocks_properties(con):
    raw_name = ibis.util.guid()
    table = con.create_table(raw_name, schema={"x": "int64"})

    try:
        assert table.schema() == ibis.schema({"x": "int64"})
        with con.raw_sql(f"SHOW CREATE TABLE `{raw_name}`") as cur:
            _, create_sql = cur.fetchone()
        assert "ENGINE=OLAP" in create_sql
        assert "DISTRIBUTED BY RANDOM" in create_sql
        assert "replication_num" in create_sql
        assert "'1'" in create_sql or '"1"' in create_sql
    finally:
        con.drop_table(raw_name, force=True)


def test_connect_from_url():
    con = ibis.connect(
        f"starrocks://{STARROCKS_USER}:{STARROCKS_PASS}@"
        f"{STARROCKS_HOST}:{STARROCKS_PORT}/{IBIS_TEST_STARROCKS_DB}"
    )
    try:
        assert con.current_database == IBIS_TEST_STARROCKS_DB
    finally:
        con.disconnect()
