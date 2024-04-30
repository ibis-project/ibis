from __future__ import annotations

import pytest

import ibis


@pytest.mark.xfail_version(pyspark=["pyspark<3.4"], reason="no catalog support")
def test_catalog_db_args(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    t = ibis.memtable({"epoch": [1712848119, 1712848121, 1712848155]})

    # create a table in specified catalog and db
    con.create_table(
        "t2", database=(con.current_catalog, "default"), obj=t, overwrite=True
    )

    assert "t2" not in con.list_tables()
    assert "t2" in con.list_tables(database="default")
    assert "t2" in con.list_tables(database="spark_catalog.default")
    assert "t2" in con.list_tables(database=("spark_catalog", "default"))

    con.drop_table("t2", database="spark_catalog.default")

    assert "t2" not in con.list_tables(database="default")


def test_create_table_no_catalog(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    t = ibis.memtable({"epoch": [1712848119, 1712848121, 1712848155]})

    # create a table in specified catalog and db
    con.create_table("t2", database=("default"), obj=t, overwrite=True)

    assert "t2" not in con.list_tables()
    assert "t2" in con.list_tables(database="default")
    assert "t2" in con.list_tables(database=("default"))

    con.drop_table("t2", database="default")

    assert "t2" not in con.list_tables(database="default")
