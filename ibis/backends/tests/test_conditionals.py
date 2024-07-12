from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

import ibis


def test_ifelse_select(backend, alltypes, df):
    table = alltypes
    table = table.select(
        [
            "int_col",
            (
                ibis.ifelse(table["int_col"] == 0, 42, -1)
                .cast("int64")
                .name("where_col")
            ),
        ]
    )

    result = table.execute()

    expected = df.loc[:, ["int_col"]].copy()

    expected["where_col"] = -1
    expected.loc[expected["int_col"] == 0, "where_col"] = 42

    backend.assert_frame_equal(result, expected)


def test_ifelse_column(backend, alltypes, df):
    expr = ibis.ifelse(alltypes["int_col"] == 0, 42, -1).cast("int64").name("where_col")
    result = expr.execute()

    expected = pd.Series(
        np.where(df.int_col == 0, 42, -1),
        name="where_col",
        dtype="int64",
    )

    backend.assert_series_equal(result, expected)


def test_substitute(backend):
    val = "400"
    t = backend.functional_alltypes
    expr = (
        t.string_col.nullif("1")
        .substitute({None: val})
        .name("subs")
        .value_counts()
        .filter(lambda t: t.subs == val)
    )
    assert expr["subs_count"].execute()[0] == t.count().execute() // 10


@pytest.mark.parametrize(
    "inp, exp",
    [
        pytest.param(
            lambda: ibis.literal(1)
            .case()
            .when(1, "one")
            .when(2, "two")
            .else_("other")
            .end(),
            "one",
            id="one_kwarg",
        ),
        pytest.param(
            lambda: ibis.literal(5).case().when(1, "one").when(2, "two").end(),
            None,
            id="fallthrough",
        ),
    ],
)
def test_value_cases_scalar(con, inp, exp):
    result = con.execute(inp())
    if exp is None:
        assert pd.isna(result)
    else:
        assert result == exp


@pytest.mark.broken(
    "exasol",
    reason="the int64 RBI column is .to_pandas()ed to an object column, which is incomparable to ints",
    raises=AssertionError,
)
def test_value_cases_column(batting):
    df = batting.to_pandas()
    expr = (
        batting.RBI.case()
        .when(5, "five")
        .when(4, "four")
        .when(3, "three")
        .else_("could be good?")
        .end()
    )
    result = expr.execute()
    expected = np.select(
        [df.RBI == 5, df.RBI == 4, df.RBI == 3],
        ["five", "four", "three"],
        "could be good?",
    )

    assert Counter(result) == Counter(expected)


def test_ibis_cases_scalar():
    expr = ibis.literal(5).case().when(5, "five").when(4, "four").end()
    result = expr.execute()
    assert result == "five"


@pytest.mark.broken(
    ["sqlite", "exasol"],
    reason="the int64 RBI column is .to_pandas()ed to an object column, which is incomparable to 5",
    raises=TypeError,
)
def test_ibis_cases_column(batting):
    t = batting
    df = batting.to_pandas()
    expr = (
        ibis.case()
        .when(t.RBI < 5, "really bad team")
        .when(t.teamID == "PH1", "ph1 team")
        .else_(t.teamID)
        .end()
    )
    result = expr.execute()
    expected = np.select(
        [df.RBI < 5, df.teamID == "PH1"],
        ["really bad team", "ph1 team"],
        df.teamID,
    )

    assert Counter(result) == Counter(expected)


@pytest.mark.broken("clickhouse", reason="special case this and returns 'oops'")
def test_value_cases_null(con):
    """CASE x WHEN NULL never gets hit"""
    e = ibis.literal(5).nullif(5).case().when(None, "oops").else_("expected").end()
    assert con.execute(e) == "expected"
