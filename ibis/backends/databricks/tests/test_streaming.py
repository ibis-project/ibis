from __future__ import annotations

import pytest
from databricks.sql import ServerOperationError

from ibis.backends.databricks.tests.conftest import DATABRICKS_CATALOG


@pytest.mark.xfail(raises=ServerOperationError, reason="no s3 directory")
def test_streaming_table_is_accessible(con):
    with con.con.cursor() as cur:
        cur.execute(f"""\
CREATE OR REFRESH STREAMING TABLE my_stream_table AS
SELECT * FROM STREAM read_files('/Volumes/{DATABRICKS_CATALOG}/default/testing_data/parquet/stream')""")
    expected = con.table("astronauts")
    streaming = con.table("my_stream_table").select(*expected.columns)
    assert streaming.schema().equals(expected.schema())
