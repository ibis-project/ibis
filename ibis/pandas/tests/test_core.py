import pytest

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

pytest.importorskip('multipledispatch')

from ibis.pandas.execution import (
    execute, execute_node, execute_first
)  # noqa: E402
from ibis.pandas.client import PandasTable, PandasClient  # noqa: E402
from ibis.pandas.core import data_preload, pre_execute  # noqa: E402
from multipledispatch.conflict import ambiguities  # noqa: E402

pytestmark = pytest.mark.pandas


@pytest.fixture
def dataframe():
    return pd.DataFrame({
        'plain_int64': list(range(1, 4)),
        'plain_strings': list('abc'),
        'dup_strings': list('dad'),
    })


@pytest.fixture
def core_client(dataframe):
    return ibis.pandas.connect({'df': dataframe})


@pytest.fixture
def ibis_table(core_client):
    return core_client.table('df')


@pytest.mark.parametrize('func', [execute, execute_node, execute_first])
def test_no_execute_ambiguities(func):
    assert not ambiguities(func.funcs)


def test_from_dataframe(dataframe, ibis_table, core_client):
    t = ibis.pandas.from_dataframe(dataframe)
    result = t.execute()
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    t = ibis.pandas.from_dataframe(dataframe, name='foo')
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    client = core_client
    t = ibis.pandas.from_dataframe(dataframe, name='foo', client=client)
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)


def test_execute_first_accepts_scope_keyword_argument(ibis_table, dataframe):

    param = ibis.param(dt.int64)

    @execute_first.register(ops.Node, pd.DataFrame)
    def foo(op, data, scope=None, **kwargs):
        assert scope is not None
        return data.dup_strings.str.len() + scope[param.op()]

    expr = ibis_table.dup_strings.length() + param
    assert expr.execute(params={param: 2}) is not None
    del execute_first.funcs[ops.Node, pd.DataFrame]
    execute_first.reorder()
    execute_first._cache.clear()


def test_data_preload(ibis_table, dataframe):
    @data_preload.register(PandasTable, pd.DataFrame)
    def data_preload_check_a_thing(_, df, **kwargs):
        return df

    result = ibis_table.execute()
    tm.assert_frame_equal(result, dataframe)

    del data_preload.funcs[PandasTable, pd.DataFrame]
    data_preload.reorder()
    data_preload._cache.clear()


def test_pre_execute_basic(ibis_table, dataframe):
    """
    Test that pre_execute has intercepted execution and provided its own
    scope dict
    """
    @pre_execute.register(ops.Node, PandasClient)
    def pre_execute_test(op, client, **kwargs):
        df = dataframe.assign(plain_int64=dataframe['plain_int64'] + 1)
        return {op: df}

    result = ibis_table.execute()
    tm.assert_frame_equal(
        result, dataframe.assign(plain_int64=dataframe['plain_int64'] + 1))

    del pre_execute.funcs[(ops.Node, PandasClient)]
    pre_execute.reorder()
    pre_execute._cache.clear()
