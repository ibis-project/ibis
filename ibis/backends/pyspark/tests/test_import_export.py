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
def test_streaming_import_not_implemented(con_streaming, method):
    with pytest.raises(NotImplementedError):
        method(con_streaming)
