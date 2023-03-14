from __future__ import annotations

import ibis
from ibis.util import guid


def test_cross_db_access(con):
    db, schema = f"tmp_db_{guid()}", f"tmp_schema_{guid()}"
    schema = f"{db}.{schema}"
    table = f"tmp_table_{guid()}"
    con.raw_sql(f"CREATE DATABASE IF NOT EXISTS {db}")
    try:
        con.raw_sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        try:
            con.raw_sql(f'CREATE TEMP TABLE {schema}."{table}" ("x" INT)')
            try:
                t = con.table(table, schema=schema)
                assert t.schema() == ibis.schema(dict(x="int"))
                assert t.execute().empty
            finally:
                con.raw_sql(f'DROP TABLE {schema}."{table}"')
        finally:
            con.raw_sql(f"DROP SCHEMA {schema}")
    finally:
        con.raw_sql(f"DROP DATABASE {db}")
