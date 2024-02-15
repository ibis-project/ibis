from __future__ import annotations

from ibis import udf


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def parse_url(string1: str, string2: str) -> str:
        ...

    expr = parse_url("http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1", "HOST")
    result = con.execute(expr)
    assert result == "facebook.com"
