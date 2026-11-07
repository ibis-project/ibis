"""Python(name) memtable injection behavior not covered by the shared suite."""

from __future__ import annotations

import ibis
from ibis.backends.chdb import _memtables


def test_memtable_self_join_reexports(mem):
    # both sides of a self join must see the full data from one registration
    t = ibis.memtable({"x": [1, 2, 3]})
    expr = t.join(t.view(), "x")
    assert mem.execute(expr.count()) == 3


def test_two_memtables_one_query(mem):
    a = ibis.memtable({"x": [1, 2]})
    b = ibis.memtable({"x": [3, 4]})
    expr = a.union(b).order_by("x")
    assert mem.execute(expr)["x"].tolist() == [1, 2, 3, 4]


def test_memtable_cleaned_up_after_finalize(mem):
    # chdb is notyet on the shared test_memtable_cleanup, so this is the only
    # coverage that finalizers unregister the injected name
    t = ibis.memtable({"x": [1, 2, 3]})
    name = t.op().name
    mem.execute(t.count())
    assert name in vars(_memtables)
    mem._make_memtable_finalizer(name)()
    assert name not in vars(_memtables)
