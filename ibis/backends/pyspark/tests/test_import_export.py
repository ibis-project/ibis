from __future__ import annotations

from operator import methodcaller

import pytest


@pytest.mark.parametrize(
    "method",
    [
        methodcaller("read_delta", "test.delta"),
        methodcaller("read_csv", "test.csv"),
        methodcaller("read_parquet", "test.parquet"),
        methodcaller("read_json", "test.json"),
    ],
)
def test_streaming_import_not_implemented(con_streaming, method):
    with pytest.raises(NotImplementedError):
        method(con_streaming)
