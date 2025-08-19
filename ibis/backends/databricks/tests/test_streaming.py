from __future__ import annotations


def test_streaming_table_is_accessible(con):
    with con.con.cursor() as cur:
        cur.execute("""\
CREATE OR REFRESH STREAMING TABLE my_stream_table AS
SELECT * FROM STREAM read_files('/Volumes/ibis_testing/default/testing_data/parquet/stream')""")
    expected = con.table("astronauts")
    streaming = con.table("my_stream_table").select(*expected.columns)
    assert streaming.schema().equals(expected.schema())
