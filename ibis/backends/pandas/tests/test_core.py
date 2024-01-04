from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.common.exceptions as com
from ibis.backends.pandas import Backend


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "plain_int64": list(range(1, 4)),
            "plain_strings": list("abc"),
            "dup_strings": list("dad"),
        }
    )


@pytest.fixture
def core_client(dataframe):
    return Backend().connect({"df": dataframe})


@pytest.fixture
def ibis_table(core_client):
    return core_client.table("df")


def test_from_dataframe(dataframe, ibis_table, core_client):
    t = Backend().from_dataframe(dataframe)
    result = t.execute()
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    t = Backend().from_dataframe(dataframe, name="foo")
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    client = core_client
    t = Backend().from_dataframe(dataframe, name="foo", client=client)
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)


def test_execute_parameter_only():
    param = ibis.param("int64")
    con = ibis.pandas.connect()
    result = con.execute(param, params={param.op(): 42})
    assert result == 42


def test_missing_data_sources():
    t = ibis.table([("a", "string")], name="t")
    expr = t.a.length()
    con = ibis.pandas.connect()
    with pytest.raises(com.UnboundExpressionError):
        con.execute(expr)


def test_unbound_table_execution():
    t = ibis.table([("a", "string")], name="t")
    expr = t.a.length()
    con = ibis.pandas.connect({"t": pd.DataFrame({"a": ["a", "ab", "abc"]})})
    result = con.execute(expr)
    assert result.tolist() == [1, 2, 3]
