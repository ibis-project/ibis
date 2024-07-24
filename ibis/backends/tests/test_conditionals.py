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
            lambda: ibis.literal(1).cases((1, "one"), (2, "two"), else_="other"),
            "one",
            id="one_kwarg",
        ),
        pytest.param(
            lambda: ibis.literal(5).cases((1, "one"), (2, "two")),
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
    expr = batting.RBI.cases(
        (5, "five"), (4, "four"), (3, "three"), else_="could be good?"
    )
    result = expr.execute()
    expected = np.select(
        [df.RBI == 5, df.RBI == 4, df.RBI == 3],
        ["five", "four", "three"],
        "could be good?",
    )

    assert Counter(result) == Counter(expected)


def test_ibis_cases_scalar():
    expr = ibis.literal(5).cases((5, "five"), (4, "four"))
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
    expr = ibis.cases(
        (t.RBI < 5, "really bad team"), (t.teamID == "PH1", "ph1 team"), else_=t.teamID
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
    e = ibis.literal(5).nullif(5).cases((None, "oops"), else_="expected")
    assert con.execute(e) == "expected"


@pytest.mark.broken("pyspark", reason="raises a ResourceWarning that we can't catch")
def test_ibis_case_is_deprecated(con):
    # just to make sure that the deprecated .case() method still works
    with pytest.warns(FutureWarning, match=".cases"):
        assert con.execute(ibis.case().when(True, "yes").end()) == "yes"
    with pytest.warns(FutureWarning, match=".cases"):
        assert pd.isna(con.execute(ibis.case().when(False, "yes").end()))
    with pytest.warns(FutureWarning, match=".cases"):
        assert con.execute(ibis.case().when(False, "yes").else_("no").end()) == "no"

    with pytest.warns(FutureWarning, match=".cases"):
        assert con.execute(ibis.literal("a").case().when("a", "yes").end()) == "yes"
    with pytest.warns(FutureWarning, match=".cases"):
        assert pd.isna(con.execute(ibis.literal("a").case().when("b", "yes").end()))
    with pytest.warns(FutureWarning, match=".cases"):
        assert (
            con.execute(ibis.literal("a").case().when("b", "yes").else_("no").end())
            == "no"
        )


@pytest.mark.parametrize(
    "inp, exp",
    [
        pytest.param(
            lambda: ibis.literal(1).cases([(1, "one"), (2, "two")], "other"),
            "one",
            id="basic",
        ),
        pytest.param(
            lambda: ibis.literal(1).cases([(1, "one"), (2, "two")], default="other"),
            "one",
            id="one_kwarg",
        ),
        pytest.param(
            lambda: ibis.literal(1).cases(
                case_result_pairs=[(1, "one"), (2, "two")], default="other"
            ),
            "one",
            id="two_kwargs",
        ),
        pytest.param(
            lambda: ibis.literal(1).cases(
                default="other", case_result_pairs=[(1, "one"), (2, "two")]
            ),
            "one",
            id="two_kwargs_swapped",
        ),
        pytest.param(
            lambda: ibis.literal(5).cases([(1, "one"), (2, "two")], "other"),
            "other",
            id="other",
        ),
        pytest.param(
            lambda: ibis.literal(5).cases([(1, "one"), (2, "two")]),
            None,
            id="fallthrough",
        ),
    ],
)
def test_value_cases_old_api_is_deprecated(con, inp, exp):
    with pytest.warns(FutureWarning):
        i = inp()
    result = con.execute(i)
    if exp is None:
        assert pd.isna(result)
    else:
        assert result == exp
