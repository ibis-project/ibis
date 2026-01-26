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
