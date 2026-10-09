from __future__ import annotations

import pandas as pd
import pandas.testing as tm

import ibis


def test_divide_precision(con):
    df = pd.DataFrame({"a": [10, 20, 30], "b": [2, 3, 4], "c": [5, 10, 15]})

    t = ibis.memtable(df)

    expr = (t.a + t.b) * t.c - t.a / t.b

    actual = con.execute(expr).squeeze()

    expected = pd.Series([55.0, 223.333, 502.5])

    tm.assert_series_equal(
        actual, expected, check_exact=False, check_names=False, atol=0.001
    )


def test_cast_string_to_float_inf_nan(con):
    # SQLite's CAST only parses a numeric prefix, so "inf" and friends used to
    # silently turn into 0.0
    df = pd.DataFrame(
        {
            "s": [
                "inf",
                "+Inf",
                " Infinity ",
                "-inf",
                "-Infinity",
                "nan",
                "-NaN",
                "1.5",
                "-2",
                None,
            ]
        }
    )
    t = ibis.memtable(df)

    result = con.execute(t.s.cast("float64").name("x"))

    inf = float("inf")
    expected = pd.Series(
        [inf, inf, inf, -inf, -inf, None, None, 1.5, -2.0, None],
        dtype="float64",
        name="x",
    )
    # SQLite can't store NaN, so NaN comes back as NULL
    tm.assert_series_equal(result, expected)
