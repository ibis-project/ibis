from __future__ import annotations

import pytest

import ibis
from ibis.util import guid


@pytest.mark.parametrize(
    ("db", "schema", "cleanup"),
    [
        (f"tmp_db_{guid()}", f"tmp_schema_{guid()}", True),
        ("ibis_testing", f"tmp_schema_{guid()}", False),
    ],
    ids=["temp", "perm"],
)
def test_cross_db_access(con, db, schema, cleanup):
    schema = f"{db}.{schema}"
    table = f"tmp_table_{guid()}"
    con.raw_sql(f"CREATE DATABASE IF NOT EXISTS {db}")
    try:
        con.raw_sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        try:
            con.raw_sql(f'CREATE TEMP TABLE {schema}.{table} ("x" INT)')
            t = con.table(table, schema=f"{schema}")
            assert t.schema() == ibis.schema(dict(x="int"))
            assert t.execute().empty
        finally:
            if cleanup:
                con.raw_sql(f"DROP SCHEMA {schema}")
    finally:
        if cleanup:
            con.raw_sql(f"DROP DATABASE {db}")
