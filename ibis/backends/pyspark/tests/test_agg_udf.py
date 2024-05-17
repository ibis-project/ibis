from __future__ import annotations

import pandas.testing as tm
import pytest

import ibis

pytest.importorskip("pyspark")


@pytest.fixture
def t(con):
    return con.table("time_indexed_table")


@pytest.fixture
def df(con):
    return con._session.table("time_indexed_table").toPandas()


@ibis.udf.agg.pandas
def avg(x) -> float:
    return x.mean()


def test_pandas_agg(t, df):
    result = t.group_by(t.key).aggregate(avg=avg(t.value)).execute()
    expected = df.groupby("key").agg(avg=("value", "mean")).reset_index()
    tm.assert_frame_equal(result, expected)
