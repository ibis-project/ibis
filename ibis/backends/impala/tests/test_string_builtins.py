from __future__ import annotations

import pytest

from ibis import literal as L
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("functional_alltypes")


@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda s: s.lower(), id="lower"),
        pytest.param(lambda s: s.upper(), id="upper"),
        pytest.param(lambda s: s.reverse(), id="reverse"),
        pytest.param(lambda s: s.strip(), id="strip"),
        pytest.param(lambda s: s.lstrip(), id="lstrip"),
        pytest.param(lambda s: s.rstrip(), id="rstrip"),
        pytest.param(lambda s: s.capitalize(), id="capitalize"),
        pytest.param(lambda s: s.length(), id="length"),
        pytest.param(lambda s: s.ascii_str(), id="ascii_str"),
        pytest.param(lambda s: s.substr(2), id="substr_2"),
        pytest.param(lambda s: s.substr(0, 3), id="substr_0_3"),
        pytest.param(lambda s: s.right(4), id="strright"),
        pytest.param(lambda s: s.like("foo%"), id="like"),
        pytest.param(lambda s: s.like(["foo%", "%bar"]), id="like_multiple"),
        pytest.param(lambda s: s.rlike(r"[\d]+"), id="rlike"),
        pytest.param(lambda s: s.re_search(r"[\d]+"), id="re_search"),
        pytest.param(lambda s: s.re_extract(r"[\d]+", 0), id="re_extract"),
        pytest.param(lambda s: s.re_replace(r"[\d]+", "aaa"), id="re_replace"),
        pytest.param(lambda s: s.repeat(2), id="repeat"),
        pytest.param(lambda s: s.host(), id="extract_host"),
        pytest.param(lambda s: s.translate("a", "b"), id="translate"),
        pytest.param(lambda s: s.find("a"), id="find"),
        pytest.param(lambda s: s.find("a", 2), id="find_with_offset"),
        pytest.param(lambda s: s.lpad(1, "a"), id="lpad_char"),
        pytest.param(lambda s: s.lpad(25), id="lpad_default"),
        pytest.param(lambda s: s.rpad(1, "a"), id="rpad_char"),
        pytest.param(lambda s: s.rpad(25), id="rpad_default"),
        pytest.param(lambda s: s.find_in_set(["a"]), id="find_in_set_single"),
        pytest.param(lambda s: s.find_in_set(["a", "b"]), id="find_in_set_multiple"),
    ],
)
def test_string_builtins(table, expr_fn, snapshot):
    expr = expr_fn(table.string_col)
    snapshot.assert_match(translate(expr), "out.sql")


def test_find(table, snapshot):
    expr = table.string_col.find("a", start=table.tinyint_col)
    snapshot.assert_match(translate(expr), "out.sql")


def test_string_join(snapshot):
    expr = L(",").join(["a", "b"])
    snapshot.assert_match(translate(expr), "out.sql")
