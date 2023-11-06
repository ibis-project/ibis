from __future__ import annotations

import ibis


def test_string_length(con):
    t = ibis.memtable({"s": ["aaa", "a", "aa"]})
    assert con.execute(t.s.length()).gt(0).all()
