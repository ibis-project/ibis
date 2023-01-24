import re

import pytest

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
from ibis import selectors as s


@pytest.fixture
def t():
    return ibis.table(
        dict(
            a="int",
            b="string",
            c="array<string>",
            d="struct<a: array<map<string, array<float>>>>",
            e="float",
            f="decimal(3, 1)",
            g="array<array<map<float, float>>>",
            ga="string",
        ),
        name="t",
    )


@pytest.mark.parametrize(
    "sel",
    [s.where(lambda _: False), s.startswith("X"), s.endswith("🙂")],
    ids=["false", "startswith", "endswith"],
)
def test_empty_selection(t, sel):
    with pytest.raises(exc.IbisError):
        t.select(sel)


def test_where(t):
    assert t.select(s.where(lambda _: True)).equals(t.select(*t.columns))


def test_numeric(t):
    assert t.select(s.numeric()).equals(t.select("a", "e", "f"))


@pytest.mark.parametrize(
    ("obj", "expected"),
    [(dt.Array, ("c", "g")), ("float", ("e",)), (dt.Decimal(3, 1), ("f",))],
    ids=["type", "string", "instance"],
)
def test_of_type(t, obj, expected):
    assert t.select(s.of_type(obj)).equals(t.select(*expected))


@pytest.mark.parametrize(
    ("prefixes", "expected"),
    [("a", ("a",)), (("a", "e"), ("a", "e"))],
    ids=["string", "tuple"],
)
def test_startswith(t, prefixes, expected):
    assert t.select(s.startswith(prefixes)).equals(t.select(*expected))


def test_endswith(t):
    assert t.select(s.endswith(("a", "d"))).equals(t.select("a", "d", "ga"))


def test_contains(t):
    assert t.select(s.contains("a")).equals(t.select("a", "ga"))


@pytest.mark.parametrize(
    ("rx", "expected"),
    [("e|f", ("e", "f")), (re.compile("e|f"), ("e", "f"))],
    ids=["string", "pattern"],
)
def test_matches(t, rx, expected):
    assert t.select(s.matches(rx)).equals(t.select(expected))


def test_compose_or(t):
    assert t.select(s.contains("a") | s.startswith("d")).equals(
        t.select("a", "d", "ga")
    )


def test_compose_and(t):
    assert t.select(s.contains("a") & s.contains("g")).equals(t.select("ga"))


def test_compose_not(t):
    assert t.select(~s.numeric()).equals(t.select("b", "c", "d", "g", "ga"))
