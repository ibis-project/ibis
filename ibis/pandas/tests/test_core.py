import pytest

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.common as com
import ibis.expr.operations as ops

pytest.importorskip('multipledispatch')

from ibis.pandas.dispatch import (
    execute_node, pre_execute, post_execute)  # noqa: E402
from ibis.pandas.client import PandasClient  # noqa: E402
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


@pytest.mark.parametrize(
    'func', [execute_node, pre_execute, post_execute]
)
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


def test_execute_parameter_only():
    param = ibis.param('int64')
    result = ibis.pandas.execute(param, params={param: 42})
    assert result == 42


def test_missing_data_sources():
    t = ibis.table([('a', 'string')])
    expr = t.a.length()
    with pytest.raises(com.UnboundExpressionError):
        ibis.pandas.execute(expr)


def test_missing_data_on_custom_client():
    class MyClient(PandasClient):
        def table(self, name):
            return ops.DatabaseTable(
                name, ibis.schema([('a', 'int64')]), self).to_expr()

    con = MyClient({})
    t = con.table('t')
    with pytest.raises(
        NotImplementedError,
        match=(
            'Could not find signature for execute_node: '
            '<DatabaseTable, MyClient>'
        )
    ):
        con.execute(t)


def test_post_execute_called_on_joins(dataframe, core_client, ibis_table):
    count = [0]

    @post_execute.register(ops.InnerJoin, pd.DataFrame)
    def tmp_left_join_exe(op, lhs, **kwargs):
        count[0] += 1
        return lhs

    left = ibis_table
    right = left.view()
    join = left.join(right, 'plain_strings')[left.plain_int64]
    result = join.execute()
    assert result is not None
    assert not result.empty
    assert count[0] == 1
