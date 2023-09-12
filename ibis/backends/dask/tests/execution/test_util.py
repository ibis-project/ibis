from __future__ import annotations

import pytest

from ibis.backends.dask.execution.util import assert_identical_grouping_keys

pytest.importorskip("dask.dataframe")


@pytest.mark.parametrize(
    "grouping, bad_grouping",
    [
        ("dup_strings", "dup_ints"),
        (["dup_strings"], ["dup_ints"]),
        (["dup_strings", "dup_ints"], ["dup_ints", "dup_strings"]),
    ],
)
def test_identical_grouping_keys_assertion(df, grouping, bad_grouping):
    gdf = df.groupby(grouping)

    a = gdf.plain_int64
    b = gdf.plain_strings

    # should not raise
    assert_identical_grouping_keys(a, b)

    c = df.groupby(bad_grouping).plain_int64

    with pytest.raises(AssertionError, match=r"Differing grouping keys passed*"):
        assert_identical_grouping_keys(a, b, c)
