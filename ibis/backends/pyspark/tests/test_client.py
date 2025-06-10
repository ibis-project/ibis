from __future__ import annotations

import pytest

import ibis
from ibis.util import gen_name


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


def test_list_databases(con):
    assert "ibis_testing" in con.list_databases(catalog="spark_catalog")


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


@pytest.mark.xfail_version(pyspark=["pyspark<3.4"], reason="no catalog support")
def test_create_table_with_partition_and_catalog(con):
    # Create a sample table with a partition column
    data = {
        "epoch": [1712848119, 1712848121, 1712848155, 1712848169],
        "category1": ["A", "B", "A", "C"],
        "category2": ["G", "J", "G", "H"],
    }

    t = ibis.memtable(data)

    # 1D partition
    table_name = "pt1"

    con.create_table(
        table_name,
        database=("spark_catalog", "default"),
        obj=t,
        overwrite=True,
        partition_by="category1",
    )
    assert table_name in con.list_tables(database="spark_catalog.default")

    partitions = (
        con.raw_sql(f"SHOW PARTITIONS spark_catalog.default.{table_name}")
        .toPandas()
        .to_dict()
    )
    expected_partitions = {
        "partition": {0: "category1=A", 1: "category1=B", 2: "category1=C"}
    }
    assert partitions == expected_partitions

    # Cleanup
    con.drop_table(table_name, database="spark_catalog.default")
    assert table_name not in con.list_tables(database="spark_catalog.default")

    # 2D partition
    table_name = "pt2"

    con.create_table(
        table_name,
        database=("spark_catalog", "default"),
        obj=t,
        overwrite=True,
        partition_by=["category1", "category2"],
    )
    assert table_name in con.list_tables(database="spark_catalog.default")

    partitions = (
        con.raw_sql(f"SHOW PARTITIONS spark_catalog.default.{table_name}")
        .toPandas()
        .to_dict()
    )
    expected_partitions = {
        "partition": {
            0: "category1=A/category2=G",
            1: "category1=B/category2=J",
            2: "category1=C/category2=H",
        }
    }
    assert partitions == expected_partitions

    # Cleanup
    con.drop_table(table_name, database="spark_catalog.default")
    assert table_name not in con.list_tables(database="spark_catalog.default")


def test_create_table_with_partition_no_catalog(con):
    data = {
        "epoch": [1712848119, 1712848121, 1712848155, 1712848169],
        "category1": ["A", "B", "A", "C"],
        "category2": ["G", "J", "G", "H"],
    }

    t = ibis.memtable(data)

    # 1D partition
    table_name = "pt1"

    con.create_table(
        table_name,
        obj=t,
        overwrite=True,
        partition_by="category1",
    )
    assert table_name in con.list_tables()

    partitions = (
        con.raw_sql(f"SHOW PARTITIONS ibis_testing.{table_name}").toPandas().to_dict()
    )
    expected_partitions = {
        "partition": {0: "category1=A", 1: "category1=B", 2: "category1=C"}
    }
    assert partitions == expected_partitions

    # Cleanup
    con.drop_table(table_name)
    assert table_name not in con.list_tables()

    # 2D partition
    table_name = "pt2"

    con.create_table(
        table_name,
        obj=t,
        overwrite=True,
        partition_by=["category1", "category2"],
    )
    assert table_name in con.list_tables()

    partitions = (
        con.raw_sql(f"SHOW PARTITIONS ibis_testing.{table_name}").toPandas().to_dict()
    )
    expected_partitions = {
        "partition": {
            0: "category1=A/category2=G",
            1: "category1=B/category2=J",
            2: "category1=C/category2=H",
        }
    }
    assert partitions == expected_partitions

    # Cleanup
    con.drop_table(table_name)
    assert table_name not in con.list_tables()


def test_insert_bug(con):
    dbname = gen_name("test_insert_bug_tb")
    name = gen_name("test_insert_bug_table")

    con.create_database(dbname, force=True)
    try:
        con.create_table(name, schema=ibis.schema([("id", "string")]), database=dbname)
        try:
            con.insert(name, ibis.memtable([{"id": "my_id"}]), database=dbname)
            con.insert(name, ibis.memtable([{"id": "my_id_2"}]), database=dbname)
        finally:
            con.drop_table(name, force=True, database=dbname)
    finally:
        con.drop_database(dbname, force=True)
