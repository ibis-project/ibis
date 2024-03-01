from __future__ import annotations

from ibis import udf


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def parse_url(string1: str, string2: str) -> str: ...

    expr = parse_url("http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1", "HOST")
    result = con.execute(expr)
    assert result == "facebook.com"


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def json_arrayagg(a) -> str:
        """Glom together some JSON."""

    ft = con.tables.functional_alltypes[:5]
    expr = json_arrayagg(ft.string_col)
    result = expr.execute()
    expected = '["0","1","2","3","4"]'
    assert result == expected
