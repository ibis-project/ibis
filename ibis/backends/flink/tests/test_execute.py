from __future__ import annotations

import ibis


def test_empty_result_with_unsupported_arrow_type(con):
    # pyflink can't build an arrow schema for an array of structs, which only
    # matters when an empty result is rebuilt from the table schema
    t = con.table("functional_alltypes")
    expr = t.filter(t.int_col < 0).select(a=ibis.array([ibis.struct({"x": t.int_col})]))
    result = expr.execute()
    assert result.empty
    assert list(result.columns) == ["a"]


def test_empty_result_has_the_same_ltz_resolution(con):
    # CURRENT_TIMESTAMP is a TIMESTAMP_LTZ(3)
    t = con.table("functional_alltypes")
    nonempty = t.select(now=ibis.now()).limit(1).execute()
    empty = t.filter(t.int_col < 0).select(now=ibis.now()).execute()
    assert empty.empty
    assert empty.dtypes["now"] == nonempty.dtypes["now"]
