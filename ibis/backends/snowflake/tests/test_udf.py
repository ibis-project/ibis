from __future__ import annotations

import sys

import pandas.testing as tm
import pytest

import ibis.expr.datatypes as dt
from ibis import _
from ibis.expr.operations import udf


def test_udf(con):
    @udf.scalar.python
    def num_vowels(s: str, include_y: bool = False) -> int:
        return sum(map(s.lower().count, "aeiou" + ("y" * include_y)))

    t = con.tables.BATTING

    expr = t.group_by(id_len=num_vowels(t.playerID)).agg(n=_.count())
    result = expr.execute()
    assert not result.empty

    expr = t.group_by(id_len=num_vowels(t.playerID, include_y=True)).agg(n=_.count())
    result = expr.execute()
    assert not result.empty


@pytest.mark.xfail(
    sys.version_info[:2] < (3, 9), reason="builtins aren't annotated in < 3.9"
)
def test_map_udf(con):
    @udf.scalar.python
    def num_vowels_map(s: str, include_y: bool = False) -> dict[str, int]:
        y = "y" * include_y
        vowels = "aeiou" + y
        counter = dict.fromkeys(vowels, 0)
        for c in s:
            if c in vowels:
                counter[c] += 1

        return counter

    t = con.tables.BATTING

    expr = t.select(vowel_dist=num_vowels_map(t.playerID))
    df = expr.execute()
    assert not df.empty


@pytest.mark.xfail(
    sys.version_info[:2] < (3, 9), reason="builtins aren't annotated in < 3.9"
)
def test_map_merge_udf(con):
    @udf.scalar.python
    def vowels_map(s: str) -> dict[str, int]:
        vowels = "aeiou"
        counter = dict.fromkeys(vowels, 0)
        for c in s:
            if c in vowels:
                counter[c] += 1

        return counter

    @udf.scalar.python
    def consonants_map(s: str) -> dict[str, int]:
        import string

        letters = frozenset(string.ascii_lowercase)
        consonants = letters - frozenset("aeiou")
        counter = dict.fromkeys(consonants, 0)

        for c in s:
            if c in consonants:
                counter[c] += 1

        return dict(counter)

    @udf.scalar.python
    def map_merge(x: dict[str, dt.json], y: dict[str, dt.json]) -> dict[str, dt.json]:
        z = x.copy()
        z.update(y)
        return z

    t = con.tables.BATTING

    expr = t.select(
        vowel_dist=map_merge(vowels_map(t.playerID), consonants_map(t.playerID))
    )
    df = expr.execute()
    assert not df.empty


def test_vectorized_udf(con):
    @udf.scalar.pandas
    def add_one(s: int) -> int:  # s is series, int is the element type
        return s + 1

    t = con.tables.BATTING
    expr = (
        t.select(year_id=lambda t: t.yearID)
        .mutate(next_year=lambda t: add_one(t.year_id))
        .order_by("year_id")
    )
    result = expr.execute()
    expected = (
        t.select(year_id=lambda t: t.yearID)
        .execute()
        .assign(next_year=lambda df: df.year_id + 1)
        .sort_values(["year_id"])
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(result, expected)
