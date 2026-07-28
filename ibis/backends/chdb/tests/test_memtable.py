"""In-memory table handling: Python(name) injection and scanning."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

import ibis
import ibis.backends.chdb as chdb_backend
from ibis import _


def test_memtable_renders_as_python_table_function(mem):
    t = ibis.memtable({"x": [1, 2, 3]})
    name = t.op().name
    sql = mem.compile(t)
    assert f"Python('{name}')" in sql


def test_memtable_from_dict(mem):
    t = ibis.memtable({"x": [1, 2, 3], "s": ["a", "b", "c"]})
    res = mem.execute(t.filter(_.x > 1).order_by("x"))
    assert res["x"].tolist() == [2, 3]
    assert res["s"].tolist() == ["b", "c"]


def test_memtable_from_pandas(mem):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0, 40.0]})
    t = ibis.memtable(df)
    assert mem.execute(t.b.sum()) == 100.0


def test_memtable_from_pyarrow(mem):
    tbl = pa.table({"g": ["x", "y", "x"], "v": [1, 2, 3]})
    t = ibis.memtable(tbl)
    res = mem.execute(t.group_by("g").agg(s=_.v.sum()).order_by("g"))
    assert res["g"].tolist() == ["x", "y"]
    assert res["s"].tolist() == [4, 2]


def test_memtable_aggregation(mem):
    t = ibis.memtable({"g": ["a", "b", "a", "b", "a"], "v": [1, 2, 3, 4, 5]})
    res = mem.execute(t.group_by("g").agg(total=_.v.sum(), n=_.count()).order_by("g"))
    assert res["total"].tolist() == [9, 6]
    assert res["n"].tolist() == [3, 2]


def test_memtable_join(mem):
    left = ibis.memtable({"k": [1, 2, 3], "l": ["a", "b", "c"]})
    right = ibis.memtable({"k": [2, 3, 4], "r": ["x", "y", "z"]})
    expr = left.join(right, "k").select("k", "l", "r").order_by("k")
    res = mem.execute(expr)
    assert res["k"].tolist() == [2, 3]
    assert res["l"].tolist() == ["b", "c"]
    assert res["r"].tolist() == ["x", "y"]


def test_memtable_self_join_reexports(mem):
    # A pandas/pyarrow-backed memtable is re-exportable, so both sides of a
    # self join see the full data.
    t = ibis.memtable({"x": [1, 2, 3]})
    expr = t.join(t.view(), "x")
    assert mem.execute(expr.count()) == 3


def test_memtable_empty(mem):
    t = ibis.memtable(pa.table({"a": pa.array([], pa.int64())}))
    assert mem.execute(t.count()) == 0


def test_two_memtables_one_query(mem):
    a = ibis.memtable({"x": [1, 2]})
    b = ibis.memtable({"x": [3, 4]})
    expr = a.union(b).order_by("x")
    assert mem.execute(expr)["x"].tolist() == [1, 2, 3, 4]


def test_memtable_cleaned_up_after_finalize(mem):
    t = ibis.memtable({"x": [1, 2, 3]})
    name = t.op().name
    mem.execute(t.count())
    assert name in vars(chdb_backend)
    mem._make_memtable_finalizer(name)()
    assert name not in vars(chdb_backend)
