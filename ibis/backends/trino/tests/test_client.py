from __future__ import annotations

import pytest

import ibis


@pytest.fixture
def tmp_name(con):
    name = ibis.util.gen_name("test_trino_table_properties")
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
