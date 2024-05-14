from __future__ import annotations

import pandas.testing as tm
import pytest

import ibis

pytest.importorskip("pyspark")


@pytest.fixture
def t(con):
    return con.table("basic_table")


@pytest.fixture
def df(con):
    return con._session.table("basic_table").toPandas()


def test_builtin_udf(t, df):
    @ibis.udf.scalar.builtin
    def repeat(x, n) -> str: ...

    result = t.mutate(repeated=repeat(t.str_col, 2)).execute()
    expected = df.assign(repeated=df.str_col * 2)
    tm.assert_frame_equal(result, expected)


def test_illegal_udf_type(t):
    with pytest.raises(
        NotImplementedError,
        match="Only Builtin UDFs and Pandas UDFs are support in the PySpark backend",
    ):

        @ibis.udf.scalar.python()
        def repeat(x, n) -> str: ...

        t.mutate(repeated=repeat(t.str_col, 2)).execute()
