from __future__ import annotations

import pytest

import ibis


@pytest.mark.xfail_version(pyspark=["pyspark<3.4"], reason="no catalog support")
def test_catalog_db_args(con):
    t = ibis.memtable({"epoch": [1712848119, 1712848121, 1712848155]})

    assert con.current_catalog == "spark_catalog"
    assert con.current_database == "ibis_testing"
    con.create_database("toot", catalog="local")

    # create a table in specified catalog and db
    con.create_table("t2", database=("local", "toot"), obj=t, overwrite=True)
    con.create_table("t3", database=("spark_catalog", "default"), obj=t, overwrite=True)

    assert con.current_database == "ibis_testing"

    assert "t2" not in con.list_tables()
    assert "t2" in con.list_tables(database="local.toot")
    assert "t2" in con.list_tables(database=("local", "toot"))

    assert "t3" not in con.list_tables()
    assert "t3" in con.list_tables(database="default")
    assert "t3" in con.list_tables(database="spark_catalog.default")

    con.drop_table("t2", database="local.toot")
    con.drop_table("t3", database="spark_catalog.default")

    assert "t2" not in con.list_tables(database="local.toot")
    assert "t3" not in con.list_tables(database="spark_catalog.default")

    con.drop_database("toot", catalog="local")

    assert con.current_catalog == "spark_catalog"
    assert con.current_database == "ibis_testing"


def test_create_table_no_catalog(con):
    t = ibis.memtable({"epoch": [1712848119, 1712848121, 1712848155]})

    assert con.current_database != "default"
    # create a table in specified catalog and db
    con.create_table("t2", database=("default"), obj=t, overwrite=True)

    assert "t2" not in con.list_tables()
    assert "t2" in con.list_tables(database="default")
    assert "t2" in con.list_tables(database=("default"))

    con.drop_table("t2", database="default")

    assert "t2" not in con.list_tables(database="default")
    assert con.current_database != "default"
