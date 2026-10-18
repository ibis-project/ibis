"""UDF behavior not covered by the shared suite."""

from __future__ import annotations

import ibis
from ibis import udf


def test_builtin_scalar_function(mem):
    # builtin ClickHouse functions need no registration (pure SQL name map)
    @udf.scalar.builtin
    def bitCount(x: int) -> int: ...

    t = ibis.memtable({"v": [7, 8]})
    res = mem.execute(t.mutate(b=bitCount(t.v)).order_by("v"))
    assert res["b"].tolist() == [3, 1]
