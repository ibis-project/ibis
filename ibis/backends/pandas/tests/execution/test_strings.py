from __future__ import annotations

from warnings import catch_warnings

import numpy as np
import pandas.testing as tm
import pytest
from pytest import param

from ibis.backends.pandas.execution.strings import sql_like_to_regex


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
        param(
            lambda s: s[s.length() - 1 :],
            lambda s: s.str[-1:],
            id="expr_slice_begin",
        ),
        param(lambda s: s[: s.length()], lambda s: s, id="expr_slice_end"),
        param(
            lambda s: s[s.length() - 2 : s.length() - 1],
            lambda s: s.str[-2:-1],
            id="expr_slice_begin_end",
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
            lambda s: s.apply(lambda x: np.array(x.split(" "))),
            id="split_spaces",
        ),
    ],
)
def test_string_ops(t, df, case_func, expected_func):
    # ignore matching UserWarnings
    with catch_warnings(record=True):
        expr = case_func(t.strings_with_space)
        result = expr.execute()
        series = expected_func(df.strings_with_space)
        tm.assert_series_equal(result, series, check_names=False)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("%abc", ".*abc"),
        ("abc%", "abc.*"),
        ("6%", "6.*"),
        ("%6%", ".*6.*"),
        ("^%6", "%6"),
        ("6^%", "6%"),
        ("6^%%", "6%.*"),
        ("^%%6", "%.*6"),
        ("^%^%6", "%%6"),
        ("6^%^%", "6%%"),
        ("6_", "6."),
        ("_6_", ".6."),
        ("^_6", "_6"),
        ("6^_", "6_"),
        ("6^__", "6_."),
        ("^__6", "_.6"),
        ("^_^_6", "__6"),
        ("6^_^_", "6__"),
        ("6%_^%_", "6.*.%."),
        ("6_^%%_", "6.%.*."),
        ("_^%%_%_^%_%_^%^__^%%^_^%%6%_", ".%.*..*.%..*.%_.%.*_%.*6.*."),
    ],
)
def test_sql_like_to_regex(pattern, expected):
    result = sql_like_to_regex(pattern, escape="^")
    assert result == f"^{expected}$"


@pytest.mark.parametrize(
    ("from_func", "to_func", "from_str", "to_str"),
    [
        param(
            lambda s: s.translate_from_strings,
            lambda s: s.translate_to_strings,
            "rmzabcghj",
            "lnsovkjfr",
            id="from_series_to_series",
        ),
        param(
            lambda s: "abc",
            lambda s: s.translate_to_strings,
            "abc",
            "ovk",
            id="from_string_to_series",
        ),
        param(
            lambda s: s.translate_from_strings,
            lambda s: "ovk",
            "abcg",
            "ovko",
            id="from_series_to_string",
        ),
    ],
)
def test_translate(
    t, df, from_func: callable, to_func: callable, from_str: str, to_str: str
):
    result = t.strings_with_space.translate(from_func(t), to_func(t)).execute()
    table = str.maketrans(from_str, to_str)
    series = df.strings_with_space.str.translate(table)
    tm.assert_series_equal(result, series, check_names=False)
