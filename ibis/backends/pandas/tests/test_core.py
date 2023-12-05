from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.common.exceptions as com
from ibis.backends.base.df.scope import Scope
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
    t = ibis.table([("a", "string")])
    expr = t.a.length()
    con = ibis.pandas.connect()
    with pytest.raises(com.OperationNotDefinedError):
        con.execute(expr)
    # TODO(kszucs): it should raise an unbound expr error
    # with pytest.raises(com.UnboundExpressionError):
    #     con.execute(expr)


def test_scope_look_up():
    # test if scope could lookup items properly
    scope = Scope()
    one_day = ibis.interval(days=1).op()
    one_hour = ibis.interval(hours=1).op()
    scope = scope.merge_scope(Scope({one_day: 1}, None))
    assert scope.get_value(one_hour) is None
    assert scope.get_value(one_day) is not None
