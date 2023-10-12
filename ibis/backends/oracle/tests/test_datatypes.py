from __future__ import annotations

import ibis


def test_failed_column_inference(con):
    # This is a table in the Docker container that we know fails
    # column type inference, so if is loaded, then we're in OK shape.
    table = con.table("ALL_DOMAINS", schema="SYS")
    assert len(table.columns)


def test_blob_raw(con):
    con.drop_table("blob_raw_blobs_blob_raw", force=True)

    with con.begin() as bind:
        bind.exec_driver_sql(
            """CREATE TABLE "blob_raw_blobs_blob_raw" ("blob" BLOB, "raw" RAW(255))"""
        )

    raw_blob = con.table("blob_raw_blobs_blob_raw")

    assert raw_blob.schema() == ibis.Schema(dict(blob="binary", raw="binary"))
