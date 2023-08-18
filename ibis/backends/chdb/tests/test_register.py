from __future__ import annotations

import pandas as pd

import ibis


def test_simple_select(con, data_dir):
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["one", "two", "three"]})
    df2 = pd.DataFrame({"c": [1, 2, 3], "d": ["①", "②", "③"]})

    t1 = ibis.memtable(df1, name="t1")
    t2 = ibis.memtable(df2, name="t2")

    r = con.execute(t1.join(t2, t1.a == t2.c))
    assert list(r.columns) == ["a", "b", "c", "d"]
    assert len(r) == 3
