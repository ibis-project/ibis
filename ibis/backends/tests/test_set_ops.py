import pandas as pd
import pytest
from pytest import param

import ibis
from ibis import _


@pytest.fixture
def union_subsets(alltypes, df):
    a = alltypes.filter((5200 <= _.id) & (_.id <= 5210))
    b = alltypes.filter((5205 <= _.id) & (_.id <= 5215))
    c = alltypes.filter((5213 <= _.id) & (_.id <= 5220))

    da = df[(5200 <= df.id) & (df.id <= 5210)]
    db = df[(5205 <= df.id) & (df.id <= 5215)]
    dc = df[(5213 <= df.id) & (df.id <= 5220)]

    return (a, b, c), (da, db, dc)


@pytest.mark.parametrize(
    "distinct",
    [param(False, id="all"), param(True, id="distinct")],
)
@pytest.mark.notimpl(["datafusion"])
def test_union(backend, union_subsets, distinct):
    (a, b, c), (da, db, dc) = union_subsets

    expr = ibis.union(a, b, c, distinct=distinct).sort_by("id")
    result = expr.execute()

    expected = (
        pd.concat([da, db, dc], axis=0)
        .sort_values("id")
        .reset_index(drop=True)
    )
    if distinct:
        expected = expected.drop_duplicates("id")

    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["datafusion"])
def test_union_mixed_distinct(backend, union_subsets):
    (a, b, c), (da, db, dc) = union_subsets

    expr = a.union(b, distinct=True).union(c, distinct=False).sort_by("id")
    result = expr.execute()
    expected = pd.concat(
        [pd.concat([da, db], axis=0).drop_duplicates("id"), dc], axis=0
    ).sort_values("id")

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            False,
            marks=pytest.mark.notyet(
                ["clickhouse", "dask", "pandas", "sqlite"],
                reason="backend doesn't support INTERSECT ALL",
            ),
            id="all",
        ),
        param(True, id="distinct"),
    ],
)
@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(["impala"])
def test_intersect(backend, alltypes, df, distinct):
    a = alltypes.filter((5200 <= _.id) & (_.id <= 5210))
    b = alltypes.filter((5205 <= _.id) & (_.id <= 5215))
    c = alltypes.filter((5195 <= _.id) & (_.id <= 5208))

    # Reset index to ensure simple RangeIndex, needed for computing `expected`
    df = df.reset_index(drop=True)
    da = df[(5200 <= df.id) & (df.id <= 5210)]
    db = df[(5205 <= df.id) & (df.id <= 5215)]
    dc = df[(5195 <= df.id) & (df.id <= 5208)]

    expr = ibis.intersect(a, b, c, distinct=distinct).sort_by("id")
    result = expr.execute()

    index = da.index.intersection(db.index).intersection(dc.index)
    expected = df.iloc[index].sort_values("id").reset_index(drop=True)
    if distinct:
        expected = expected.drop_duplicates()

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            False,
            marks=pytest.mark.notyet(
                ["clickhouse", "dask", "pandas", "sqlite"],
                reason="backend doesn't support EXCEPT ALL",
            ),
            id="all",
        ),
        param(True, id="distinct"),
    ],
)
@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(["impala"])
def test_difference(backend, alltypes, df, distinct):
    a = alltypes.filter((5200 <= _.id) & (_.id <= 5210))
    b = alltypes.filter((5205 <= _.id) & (_.id <= 5215))
    c = alltypes.filter((5195 <= _.id) & (_.id <= 5202))

    # Reset index to ensure simple RangeIndex, needed for computing `expected`
    df = df.reset_index(drop=True)
    da = df[(5200 <= df.id) & (df.id <= 5210)]
    db = df[(5205 <= df.id) & (df.id <= 5215)]
    dc = df[(5195 <= df.id) & (df.id <= 5202)]

    expr = ibis.difference(a, b, c, distinct=distinct).sort_by("id")
    result = expr.execute()

    index = da.index.difference(db.index).difference(dc.index)
    expected = df.iloc[index].sort_values("id").reset_index(drop=True)
    if distinct:
        expected = expected.drop_duplicates()

    backend.assert_frame_equal(result, expected)
