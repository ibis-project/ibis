from __future__ import annotations

import pytest
import sqlglot as sg
import sqlglot.expressions as sge

from ibis.backends.sql import drop_statement


@pytest.mark.parametrize(
    ("dialect", "kind", "this"),
    [
        ("postgres", "VIEW", sg.table("ibis_tmp")),
        ("postgres", "SCHEMA", sg.table("myschema")),
        ("duckdb", "VIEW", "my_temp_view"),
        ("duckdb", "TABLE", sg.table("final_table")),
        ("sqlite", "TABLE", "ibis_table"),
        ("snowflake", "DATABASE", sg.to_identifier("mydb")),
        ("trino", "SCHEMA", sg.table("myschema", catalog="cat")),
    ],
)
@pytest.mark.parametrize("exists", [True, False])
def test_drop_statement_preserves_target(dialect, kind, this, exists):
    # sqlglot 30 renamed `Drop.this` to `Drop.tables` and silently ignores
    # unknown kwargs, which previously rendered a nameless `DROP ... IF EXISTS`
    stmt = drop_statement(kind=kind, this=this, exists=exists)
    sql = stmt.sql(dialect)
    target = this.sql(dialect) if isinstance(this, sge.Expression) else this
    assert "DROP" in sql
    assert kind in sql
    assert target.strip("'\"") in sql


def test_drop_statement_forwards_extra_kwargs():
    stmt = drop_statement(
        kind="TABLE", this=sg.to_identifier("t"), exists=True, cascade=True
    ).sql("postgres")
    assert "CASCADE" in stmt
