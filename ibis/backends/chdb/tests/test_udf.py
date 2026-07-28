"""User-defined and builtin scalar function support."""

from __future__ import annotations

import pytest

import ibis
from ibis import udf


def test_python_scalar_udf(mem):
    @udf.scalar.python
    def add_one(x: int) -> int:
        return x + 1

    t = ibis.memtable({"v": [10, 20, 30]})
    res = mem.execute(t.mutate(w=add_one(t.v)).order_by("v"))
    assert res["w"].tolist() == [11, 21, 31]


def test_python_scalar_udf_multiple_args(mem):
    @udf.scalar.python
    def combine(a: int, b: float) -> float:
        return a + b

    t = ibis.memtable({"a": [1, 2], "b": [0.5, 1.5]})
    res = mem.execute(t.mutate(c=combine(t.a, t.b)).order_by("a"))
    assert res["c"].tolist() == [1.5, 3.5]


def test_python_scalar_udf_string(mem):
    @udf.scalar.python
    def shout(s: str) -> str:
        return s.upper()

    t = ibis.memtable({"s": ["a", "bc"]})
    res = mem.execute(t.mutate(u=shout(t.s)).order_by("s"))
    assert res["u"].tolist() == ["A", "BC"]


def test_builtin_scalar_function(mem):
    # Builtin ClickHouse functions need no registration (pure SQL name map).
    @udf.scalar.builtin
    def bitCount(x: int) -> int: ...

    t = ibis.memtable({"v": [7, 8]})
    res = mem.execute(t.mutate(b=bitCount(t.v)).order_by("v"))
    assert res["b"].tolist() == [3, 1]


def test_unsupported_udf_type_raises(mem):
    import ibis.common.exceptions as com

    @udf.scalar.python
    def bad(x: list[int]) -> int:
        return sum(x)

    t = ibis.memtable({"v": [1, 2]})
    with pytest.raises((com.UnsupportedBackendType, Exception)):
        mem.execute(t.mutate(w=bad(t.v)))
