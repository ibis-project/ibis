"""UDF behavior not covered by the shared suite."""

from __future__ import annotations

import datetime

import ibis
import ibis.expr.datatypes as dt
from ibis import udf


def test_builtin_scalar_function(mem):
    # builtin ClickHouse functions need no registration (pure SQL name map)
    @udf.scalar.builtin
    def bitCount(x: int) -> int: ...

    t = ibis.memtable({"v": [7, 8]})
    res = mem.execute(t.mutate(b=bitCount(t.v)).order_by("v"))
    assert res["b"].tolist() == [3, 1]


def test_timestamp_udf_with_fixed_offset_timezone(mem):
    # ClickHouse can't parse a DateTime('+05:00'), so the UDF is typed as UTC
    @udf.scalar.python(signature=((dt.int64,), dt.Timestamp(timezone="+05:00")))
    def f(x):  # noqa: ARG001
        return datetime.datetime(2026, 7, 22, 1, 2, 3, tzinfo=datetime.timezone.utc)

    assert str(mem.execute(f(ibis.literal(1)))) == "2026-07-22 06:02:03+05:00"
