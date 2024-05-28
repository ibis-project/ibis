from __future__ import annotations

from operator import methodcaller

import pytest


@pytest.mark.parametrize(
    "method",
    [
        methodcaller("read_delta", path="test.delta"),
        methodcaller("read_csv", source_list="test.csv"),
        methodcaller("read_parquet", path="test.parquet"),
        methodcaller("read_json", source_list="test.json"),
    ],
)
@pytest.mark.notyet(["pyspark"], raises=NotImplementedError)
def test_streaming_import_not_implemented(con_streaming, method):
    method(con_streaming)
