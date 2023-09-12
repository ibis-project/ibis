from __future__ import annotations

import re

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.dataframe.utils import tm
from pytest import param

import ibis
import ibis.expr.operations as ops


def make_dask_data_frame(npartitions):
    df = pd.DataFrame(np.random.randn(30, 4), columns=list("ABCD"))
    return dd.from_pandas(df, npartitions=npartitions)


@pytest.fixture
def client(npartitions):
    return ibis.dask.connect(
        {
            "df": dd.from_pandas(
                pd.DataFrame({"a": [1, 2, 3], "b": list("abc")}),
                npartitions=npartitions,
            ),
            "df_unknown": dd.from_pandas(
                pd.DataFrame({"array_of_strings": [["a", "b"], [], ["c"]]}),
                npartitions=npartitions,
            ),
        }
    )


@pytest.fixture
def table(client):
    return client.table("df")


def test_connect_no_args():
    con = ibis.dask.connect()
    assert dict(con.tables) == {}


def test_client_table(table):
    assert isinstance(table.op(), ops.DatabaseTable)


def test_create_table(client, npartitions):
    ddf = make_dask_data_frame(npartitions)
    client.create_table("testing", obj=ddf)
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


def test_invalid_connection_parameter_types(npartitions):
    # Check that the user receives a TypeError with an informative message when
    # passing invalid an connection parameter to the backend.
    expected_msg = re.escape(
        "Expected an instance of 'dask.dataframe.DataFrame' for 'invalid_str',"
        " got an instance of 'str' instead."
    )
    with pytest.raises(TypeError, match=expected_msg):
        ibis.dask.connect(
            {
                "valid_dask_df": dd.from_pandas(
                    pd.DataFrame({"a": [1, 2, 3], "b": list("abc")}),
                    npartitions=npartitions,
                ),
                "valid_pandas_df": pd.DataFrame({"a": [1, 2, 3], "b": list("abc")}),
                "invalid_str": "file.csv",
            }
        )

    expeced_msg = re.escape(
        "Expected an instance of 'dask.dataframe.DataFrame' for 'df', "
        "got an instance of 'str' instead."
    )
    con = ibis.dask.connect()
    with pytest.raises(TypeError, match=expeced_msg):
        con.from_dataframe("file.csv")
