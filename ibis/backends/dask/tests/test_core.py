import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.pandas.dispatch import execute_node as pandas_execute_node
from ibis.expr.scope import Scope

dd = pytest.importorskip("dask.dataframe")

from dask.dataframe.utils import tm  # noqa: E402

from ibis.backends.dask import Backend  # noqa: E402
from ibis.backends.dask.core import execute  # noqa: E402
from ibis.backends.dask.dispatch import (  # noqa: E402
    execute_node,
    post_execute,
    pre_execute,
)


def test_from_dataframe(dataframe, ibis_table, core_client):
    t = ibis.dask.from_dataframe(dataframe)
    result = t.execute()
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    t = ibis.dask.from_dataframe(dataframe, name='foo')
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    client = core_client
    t = ibis.dask.from_dataframe(dataframe, name='foo', client=client)
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)


def test_pre_execute_basic():
    """Test that pre_execute has intercepted execution and provided its own
    scope dict."""

    @pre_execute.register(ops.Add)
    def pre_execute_test(op, *clients, scope=None, **kwargs):
        return Scope({op: 4}, None)

    one = ibis.literal(1)
    expr = one + one
    result = execute(expr.op())
    assert result == 4

    del pre_execute.funcs[(ops.Add,)]
    pre_execute.reorder()
    pre_execute._cache.clear()


def test_execute_parameter_only():
    param = ibis.param('int64')
    result = execute(param.op(), params={param.op(): 42})
    assert result == 42


def test_missing_data_sources():
    t = ibis.table([('a', 'string')])
    expr = t.a.length()
    with pytest.raises(com.UnboundExpressionError):
        execute(expr.op())


def test_missing_data_on_custom_client():
    class MyBackend(Backend):
        def table(self, name):
            return ops.DatabaseTable(
                name, ibis.schema([('a', 'int64')]), self
            ).to_expr()

    con = MyBackend()
    t = con.table('t')
    with pytest.raises(
        NotImplementedError,
        match='Could not find signature for execute_node: <DatabaseTable, MyBackend>',
    ):
        con.execute(t)


def test_post_execute_called_on_joins(dataframe, core_client, ibis_table):
    count = [0]

    @post_execute.register(ops.InnerJoin, dd.DataFrame)
    def tmp_left_join_exe(op, lhs, **kwargs):
        count[0] += 1
        return lhs

    left = ibis_table
    right = left.view()
    join = left.join(right, 'plain_strings')[left.plain_int64]
    result = join.execute()
    assert result is not None
    assert len(result.index) > 0
    assert count[0] == 1


def test_scope_look_up():
    # test if scope could lookup items properly
    scope = Scope()
    one_day = ibis.interval(days=1).op()
    one_hour = ibis.interval(hours=1).op()
    scope = scope.merge_scope(Scope({one_day: 1}, None))
    assert scope.get_value(one_hour) is None
    assert scope.get_value(one_day) is not None


def test_new_dispatcher():
    types = (ops.TableColumn, dd.DataFrame)
    assert execute_node.dispatch(*types) is not None
    assert pandas_execute_node.dispatch(*types) is None
