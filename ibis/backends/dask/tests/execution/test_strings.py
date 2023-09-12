from __future__ import annotations

from warnings import catch_warnings

import pytest
from pytest import param

dd = pytest.importorskip("dask.dataframe")
from dask.dataframe.utils import tm  # noqa: E402


@pytest.mark.parametrize(
    ("case_func", "expected_func"),
    [
        param(
            lambda s: s.length(),
            lambda s: s.str.len().astype("int32"),
            id="length",
        ),
        param(lambda s: s.substr(1, 2), lambda s: s.str[1:3], id="substr"),
        param(lambda s: s[1:3], lambda s: s.str[1:3], id="slice"),
        # TODO - execute_substring_series_series is broken
        param(
            lambda s: s[s.length() - 1 :],
            lambda s: s.str[-1:],
            id="expr_slice_begin",
            marks=pytest.mark.xfail,
        ),
        param(
            lambda s: s[: s.length()],
            lambda s: s,
            id="expr_slice_end",
            marks=pytest.mark.xfail,
        ),
        param(
            lambda s: s[s.length() - 2 : s.length() - 1],
            lambda s: s.str[-2:-1],
            id="expr_slice_begin_end",
            marks=pytest.mark.xfail,
        ),
        param(lambda s: s.strip(), lambda s: s.str.strip(), id="strip"),
        param(lambda s: s.lstrip(), lambda s: s.str.lstrip(), id="lstrip"),
        param(lambda s: s.rstrip(), lambda s: s.str.rstrip(), id="rstrip"),
        param(
            lambda s: s.lpad(3, "a"),
            lambda s: s.str.pad(3, side="left", fillchar="a"),
            id="lpad",
        ),
        param(
            lambda s: s.rpad(3, "b"),
            lambda s: s.str.pad(3, side="right", fillchar="b"),
            id="rpad",
        ),
        param(lambda s: s.reverse(), lambda s: s.str[::-1], id="reverse"),
        param(lambda s: s.lower(), lambda s: s.str.lower(), id="lower"),
        param(lambda s: s.upper(), lambda s: s.str.upper(), id="upper"),
        param(
            lambda s: s.capitalize(),
            lambda s: s.str.capitalize(),
            id="capitalize",
        ),
        param(lambda s: s.repeat(2), lambda s: s * 2, id="repeat"),
        param(
            lambda s: s.contains("a"),
            lambda s: s.str.contains("a", regex=False),
            id="contains",
        ),
        param(
            lambda s: ~(s.contains("a")),
            lambda s: ~s.str.contains("a", regex=False),
            id="not_contains",
        ),
        param(
            lambda s: s.like("a"),
            lambda s: s.str.contains("^a$", regex=True),
            id="like",
        ),
        param(
            lambda s: s.re_search("(ab)+"),
            lambda s: s.str.contains("(?:ab)+", regex=True),
            id="re_search",
        ),
        param(
            lambda s: s.re_search("(ab)+") | s.re_search("d{1,2}ee"),
            lambda s: (
                s.str.contains("(?:ab)+", regex=True) | s.str.contains("d{1,2}ee")
            ),
            id="re_search_or",
        ),
        param(
            lambda s: s + s.rpad(3, "a"),
            lambda s: s + s.str.pad(3, side="right", fillchar="a"),
            id="rpad2",
        ),
        param(
            lambda s: s.split(" "),
            lambda s: s.str.split(" "),
            id="split_spaces",
            marks=pytest.mark.notimpl(["dask"], reason="arrays - #2553"),
        ),
    ],
)
def test_string_ops(t, df, case_func, expected_func):
    # ignore matching UserWarnings
    with catch_warnings(record=True):
        expr = case_func(t.strings_with_space)
        result = expr.compile()
        series = expected_func(df.strings_with_space)
        tm.assert_series_equal(result.compute(), series.compute(), check_index=False)


def test_grouped_string_re_search(t, df):
    expr = t.group_by(t.dup_strings).aggregate(
        sum=t.strings_with_space.re_search("(ab)+").cast("int64").sum()
    )

    result = expr.compile()
    expected = (
        df.groupby("dup_strings")
        .strings_with_space.apply(lambda s: s.str.contains("(?:ab)+", regex=True).sum())
        .reset_index()
        .rename(columns={"strings_with_space": "sum"})
    )

    tm.assert_frame_equal(result.compute(), expected.compute())
