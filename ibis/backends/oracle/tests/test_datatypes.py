from __future__ import annotations

import pytest
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.sql.datatypes import OracleType


def test_failed_column_inference(con):
    # This is a table in the Docker container that we know fails
    # column type inference, so if is loaded, then we're in OK shape.
    table = con.table("ALL_DOMAINS", database="SYS")
    assert len(table.columns)


def test_blob_raw(con):
    con.drop_table("blob_raw_blobs_blob_raw", force=True)

    with con.begin() as bind:
        bind.execute(
            """CREATE TABLE "blob_raw_blobs_blob_raw" ("blob" BLOB, "raw" RAW(255))"""
        )

    raw_blob = con.table("blob_raw_blobs_blob_raw")

    assert raw_blob.schema() == ibis.Schema(dict(blob="binary", raw="binary"))


@pytest.mark.parametrize(
    ("typ", "length"),
    [("VARCHAR(4000)", None), ("VARCHAR(3)", 3), ("VARCHAR(4000)", 4000)],
)
def test_string(typ, length):
    expected = sg.parse_one(typ, read="oracle", into=sge.DataType)
    result = OracleType.from_ibis(dt.String(length=length))
    assert result == expected


def test_number(con):
    con.drop_table("number_table", force=True)

    with con.begin() as bind:
        bind.execute(
            """CREATE TABLE "number_table" ("number_8_2" NUMBER(8, 2), "number_8" NUMBER(8), "number_default" NUMBER)"""
        )

    raw_blob = con.table("number_table")

    assert raw_blob.schema() == ibis.Schema(
        dict(number_8_2="decimal(8, 2)", number_8="int64", number_default="int64")
    )
