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


@ibis.udf.scalar.builtin
def repeat(x, n) -> str: ...


def test_builtin_udf(t, df):
    result = t.mutate(repeated=repeat(t.str_col, 2)).execute()
    expected = df.assign(repeated=df.str_col * 2)
    tm.assert_frame_equal(result, expected)


def test_illegal_udf_type(t):
    @ibis.udf.scalar.pyarrow
    def my_add_one(x) -> str:
        import pyarrow.compute as pac

        return pac.add(pac.binary_length(x), 1)

    expr = t.select(repeated=my_add_one(t.str_col))

    with pytest.raises(
        NotImplementedError,
        match="Only Builtin UDFs and Pandas UDFs are supported in the PySpark backend",
    ):
        expr.execute()
