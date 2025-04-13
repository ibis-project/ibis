from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyspark.errors.exceptions.captured import AnalysisException

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


@pytest.mark.parametrize(
    "database_param",
    [
        pytest.param(
            ("spark_catalog", "ibis_testing"),
            marks=pytest.mark.xfail_version(
                condition=["pyspark<3.4"], reason="no catalog support"
            ),
            id="with_catalog",
        ),
        pytest.param(None, id="no_catalog"),
    ],
)
def test_create_table_with_partitions(con, database_param):
    # Create a sample table with a partition column
    data = {
        "epoch": [1712848119, 1712848121, 1712848155, 1712848169],
        "category1": ["A", "B", "A", "C"],
        "category2": ["G", "J", "G", "H"],
    }

    t = ibis.memtable(data)

    db_ref = database_param
    db_str = "spark_catalog.ibis_testing" if db_ref else None

    # 1D partition
    table_name = "pt1"

    con.create_table(
        table_name,
        database=db_ref,
        obj=t,
        overwrite=True,
        partitionBy="category1",
    )
    assert table_name in con.list_tables(database=db_str)

    loc = (
        f"spark_catalog.ibis_testing.{table_name}"
        if db_ref
        else f"ibis_testing.{table_name}"
    )
    partitions = con.raw_sql(f"SHOW PARTITIONS {loc}").toPandas().to_dict()
    expected_partitions = {
        "partition": {0: "category1=A", 1: "category1=B", 2: "category1=C"}
    }
    assert partitions == expected_partitions

    # Cleanup
    con.drop_table(table_name, database=db_str)
    assert table_name not in con.list_tables(database=db_str)

    # 2D partition
    table_name = "pt2"

    con.create_table(
        table_name,
        database=db_ref,
        obj=t,
        overwrite=True,
        partitionBy=["category1", "category2"],
    )
    assert table_name in con.list_tables(database=db_str)

    loc = (
        f"spark_catalog.ibis_testing.{table_name}"
        if db_ref
        else f"ibis_testing.{table_name}"
    )
    partitions = con.raw_sql(f"SHOW PARTITIONS {loc}").toPandas().to_dict()
    expected_partitions = {
        "partition": {
            0: "category1=A/category2=G",
            1: "category1=B/category2=J",
            2: "category1=C/category2=H",
        }
    }
    assert partitions == expected_partitions

    # Cleanup
    con.drop_table(table_name, database=db_str)
    assert table_name not in con.list_tables(database=db_str)


@pytest.mark.parametrize(
    "format",
    ["delta", "parquet", "csv"],
)
@pytest.mark.parametrize(
    "database_param",
    [
        pytest.param(
            ("spark_catalog", "ibis_testing"),
            marks=pytest.mark.xfail_version(
                condition=["pyspark<3.4"], reason="no catalog support"
            ),
            id="with_catalog",
        ),
        pytest.param(None, id="no_catalog"),
    ],
)
def test_create_table_kwargs(con, format, database_param):
    def compare_tables(t_out, t_in):
        cols = list(t_out.columns)
        expected = t_out[cols].sort_values(cols).reset_index(drop=True)
        result = t_in[cols].sort_values(cols).reset_index(drop=True)
        assert_frame_equal(expected, result)

    base_data = {
        "epoch": [1712848119, 1712848121, 1712848155, 1712848169],
        "category1": ["A", "B", "A", "C"],
    }

    table_name = f"kwarg_test_{format}"
    db_ref = database_param
    db_str = "spark_catalog.ibis_testing" if db_ref else None

    # Helper to get table
    def get_table():
        return con.table(table_name, database=db_ref).to_pandas()

    # 1. Create db table
    t = ibis.memtable(base_data)
    con.create_table(
        table_name,
        database=db_ref,
        obj=t,
        overwrite=True,
        partitionBy="category1",
        format=format,
    )

    assert table_name in con.list_tables(database=db_str)
    compare_tables(t.to_pandas(), get_table())

    # 2. Append, same schema (mode & format kwargs) - this is similar behavior to `insert` method
    con.create_table(
        table_name,
        database=db_ref,
        obj=t,
        overwrite=True,  # Will get overwritten by mode
        mode="append",
        partitionBy="category1",
        format=format,
    )

    assert table_name in con.list_tables(database=db_str)
    expected_2x = pd.concat([t.to_pandas()] * 2, ignore_index=True)
    compare_tables(expected_2x, get_table())

    # 3. Overwrite table & schema (mode, overwriteSchema, & format kwargs)
    data2 = {
        **base_data,
        "category2": ["G", "J", "G", "H"],
    }
    t2 = ibis.memtable(data2)

    con.create_table(
        table_name,
        database=db_ref,
        obj=t2,
        mode="overwrite",
        overwriteSchema=True,
        format=format,
    )

    assert table_name in con.list_tables(database=db_str)
    compare_tables(t2.to_pandas(), get_table())

    # 4. Append and merge schema (mode, mergeSchema, & format kwargs)
    data_merge = {**data2, "category3": ["W", "Z", "Q", "X"]}
    t_merge = ibis.memtable(data_merge)

    if format == "delta":
        con.create_table(
            table_name,
            database=db_ref,
            obj=t_merge,
            mode="append",
            mergeSchema=True,
            format=format,
        )

        assert table_name in con.list_tables(database=db_str)
        expected_merged = pd.concat(
            [t2.to_pandas(), t_merge.to_pandas()], ignore_index=True
        ).fillna(value=pd.NA)

        compare_tables(expected_merged, get_table().fillna(value=pd.NA))
    else:  # Not supported on write for parquet or csv
        with pytest.raises(AnalysisException) as _:
            con.create_table(
                table_name,
                database=db_ref,
                obj=t_merge,
                mode="append",
                mergeSchema=True,
                format=format,
            )

    # Cleanup
    con.drop_table(table_name, database=db_str)
    assert table_name not in con.list_tables()
