from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from pytest import param

import ibis
import ibis.expr.operations as ops


@pytest.fixture
def client():
    return ibis.pandas.connect(
        {
            "df": pd.DataFrame({"a": [1, 2, 3], "b": list("abc")}),
            "df_unknown": pd.DataFrame({"array_of_strings": [["a", "b"], [], ["c"]]}),
        }
    )


@pytest.fixture
def table(client):
    return client.table("df")


@pytest.fixture
def test_data():
    test_data = test_data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": list("abcde")})
    return test_data


def test_connect_no_args():
    con = ibis.pandas.connect()
    assert dict(con.tables) == {}


def test_client_table(table):
    assert isinstance(table.op(), ops.DatabaseTable)


@pytest.mark.parametrize(
    "lamduh",
    [(lambda df: df), (lambda df: pa.Table.from_pandas(df))],
    ids=["dataframe", "pyarrow table"],
)
def test_create_table(client, test_data, lamduh):
    test_data = lamduh(test_data)
    client.create_table("testing", obj=test_data)
    assert "testing" in client.list_tables()
    client.create_table("testingschema", schema=client.get_schema("testing"))
    assert "testingschema" in client.list_tables()


def test_literal(client):
    lit = ibis.literal(1)
    result = client.execute(lit)
    assert result == 1


def test_list_tables(client):
    assert client.list_tables(like="df_unknown")
    assert not client.list_tables(like="not_in_the_database")
    assert client.list_tables()


def test_drop(table):
    table = table.mutate(c=table.a)
    expr = table.drop("a")
    result = expr.execute()
    expected = table[["b", "c"]].execute()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "unit",
    [
        "Y",
        "M",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        param("ps", marks=pytest.mark.xfail),
        param("fs", marks=pytest.mark.xfail),
        param("as", marks=pytest.mark.xfail),
    ],
)
def test_datetime64_infer(client, unit):
    value = np.datetime64("2018-01-02", unit)
    expr = ibis.literal(value, type="timestamp")
    result = client.execute(expr)
    assert result == pd.Timestamp(value).to_pydatetime()
