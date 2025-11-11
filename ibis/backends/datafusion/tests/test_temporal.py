from __future__ import annotations

from operator import methodcaller

import pytest
from pytest import param

import ibis
from ibis.backends.datafusion.tests.conftest import BackendTest

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        param(
            methodcaller("hour"),
            14,
            id="hour",
        ),
        param(
            methodcaller("minute"),
            48,
            id="minute",
        ),
        param(
            methodcaller("second"),
            5,
            id="second",
        ),
        param(
            methodcaller("millisecond"),
            359,
            id="millisecond",
        ),
    ],
)
def test_time_extract_literal(con, func, expected):
    value = ibis.time("14:48:05.359")
    assert con.execute(func(value).name("tmp")) == expected


@pytest.mark.parametrize(
    "pattern",
    [
        "%Y-%m-%d",
        "%Y:%m:%d",
        "%Y%m%d",
        "%d-%m-%Y",
        "%Y-%b-%d",
        "%Y-%B-%d",
        "%Y-%B-%d-%a",
        "%F",
        "%Y:%m:%d:%H:%M:%S",
    ],
)
def test_strftime(con, pattern):
    df = pd.DataFrame({"time_col": pa.array([18506, 18507, 18508, 18509], pa.date32())})

    t = ibis.memtable(df)

    name = "formatted"
    expr = t.time_col.strftime(pattern).name(name)
    expected = df.time_col.dt.strftime(pattern).rename(name)

    result = con.execute(expr)
    BackendTest.assert_series_equal(result, expected)
