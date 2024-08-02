from __future__ import annotations

import pandas.testing as tm
import pytest

import ibis
from ibis.expr.tests.snapshots.test_sql.test_parse_sql_aggregation_with_multiple_joins.decompiled import \
    result

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

def test_pyarrow_udf(t, df):
    result = t.mutate(repeated=pyarrow_repeat(t.str_col, 2)).execute()
    expected = df.assign(repeated=df.str_col * 2)
    tm.assert_frame_equal(result, expected)


