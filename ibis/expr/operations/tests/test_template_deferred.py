from __future__ import annotations

import ibis
from ibis.tstring import t


def test_repr():
    five = 5  # noqa: F841
    d = ibis.sql_value(t("{ibis._.foo + 3} * {five}"))
    r = repr(d)
    expected = """TemplateValueResolver(template='(_.foo + 3) * 5', dialect='duckdb', dtype=None)"""
    assert r == expected
