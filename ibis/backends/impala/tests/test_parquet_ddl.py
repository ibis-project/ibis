from __future__ import annotations

from posixpath import join as pjoin

import pytest

import ibis
from ibis.tests.util import assert_equal

pytest.importorskip("impala")

from impala.error import HiveServer2Error  # noqa: E402


def test_parquet_file_with_name(con, test_data_dir, temp_table):
    hdfs_path = pjoin(test_data_dir, "impala/parquet/region")

    name = temp_table
    schema = ibis.schema(
        [
            ("r_regionkey", "int16"),
            ("r_name", "string"),
            ("r_comment", "string"),
        ]
    )
    con.parquet_file(hdfs_path, schema=schema, name=temp_table)

    # table still exists
    con.table(name)


def test_query_parquet_file_with_schema(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, "impala/parquet/region")

    schema = ibis.schema(
        [
            ("r_regionkey", "int16"),
            ("r_name", "string"),
            ("r_comment", "string"),
        ]
    )

    table = con.parquet_file(hdfs_path, schema=schema)

    name = table._qualified_name

    # table exists
    con.table(name)

    expr = table.r_name.value_counts()
    expr.execute()

    assert table.count().execute() == 5


def test_query_parquet_file_like_table(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, "impala/parquet/region")

    ex_schema = ibis.schema(
        [
            ("r_regionkey", "int32"),
            ("r_name", "string"),
            ("r_comment", "string"),
        ]
    )

    table = con.parquet_file(hdfs_path, like_table="region")

    assert_equal(table.schema(), ex_schema)


def test_query_parquet_infer_schema(con, test_data_dir):
    hdfs_path = pjoin(test_data_dir, "impala/parquet/region")
    table = con.parquet_file(hdfs_path, like_table="region")

    # NOTE: the actual schema should have an int16, but bc this is being
    # inferred from a parquet file, which has no notion of int16, the
    # inferred schema will have an int32 instead.
    ex_schema = ibis.schema(
        [
            ("r_regionkey", "int32"),
            ("r_name", "string"),
            ("r_comment", "string"),
        ]
    )

    assert_equal(table.schema(), ex_schema)


def test_create_table_persist_fails_if_called_twice(con, temp_table, test_data_dir):
    hdfs_path = pjoin(test_data_dir, "impala/parquet/region")
    con.parquet_file(hdfs_path, like_table="region", name=temp_table)

    with pytest.raises(HiveServer2Error):
        con.parquet_file(hdfs_path, like_table="region", name=temp_table)
