from __future__ import annotations

import pandas.testing as tm
import pytest

import ibis
from ibis.backends.pyspark import PYSPARK_LT_35
from ibis.conftest import IS_SPARK_REMOTE

pytest.importorskip("pyspark")


@pytest.fixture
def t(con):
    return con.table("basic_table")


@pytest.fixture
def df(con):
    return con._session.table("basic_table").toPandas()


@ibis.udf.scalar.builtin
def repeat(x, n) -> str: ...


@ibis.udf.scalar.python
def py_repeat(x: str, n: int) -> str:
    return x * n


@ibis.udf.scalar.pyarrow
def pyarrow_repeat(x: str, n: int) -> str:
    return x * n


def test_builtin_udf(t, df):
    result = t.mutate(repeated=repeat(t.str_col, 2)).execute()
    expected = df.assign(repeated=df.str_col * 2)
    tm.assert_frame_equal(result, expected)


def test_python_udf(t, df):
    result = t.mutate(repeated=py_repeat(t.str_col, 2)).execute()
    expected = df.assign(repeated=df.str_col * 2)
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(PYSPARK_LT_35, reason="pyarrow UDFs require PySpark 3.5+")
@pytest.mark.xfail(
    IS_SPARK_REMOTE,
    reason="pyarrow UDFs aren't tested with spark remote due to environment setup complexities",
)
def test_pyarrow_udf(t, df):
    result = t.mutate(repeated=pyarrow_repeat(t.str_col, 2)).execute()
    expected = df.assign(repeated=df.str_col * 2)
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(not PYSPARK_LT_35, reason="pyarrow UDFs require PySpark 3.5+")
def test_illegal_udf_type(t):
    @ibis.udf.scalar.pyarrow
    def my_add_one(x) -> str:
        import pyarrow.compute as pac

        return pac.add(pac.binary_length(x), 1)

    expr = t.select(repeated=my_add_one(t.str_col))

    with pytest.raises(
        NotImplementedError,
        match="pyarrow UDFs are only supported in pyspark >= 3.5",
    ):
        expr.execute()
