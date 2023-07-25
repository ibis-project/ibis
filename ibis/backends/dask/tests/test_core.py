from __future__ import annotations

import pytest
from dask.dataframe.utils import tm

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.df.scope import Scope
from ibis.backends.pandas.dispatch import execute_node as pandas_execute_node

dd = pytest.importorskip("dask.dataframe")
import pandas as pd  # noqa: E402

from ibis.backends.dask.core import execute  # noqa: E402
from ibis.backends.dask.dispatch import (  # noqa: E402
    execute_node,
    post_execute,
    pre_execute,
)

dd = pytest.importorskip("dask.dataframe")


def test_table_from_dataframe(dataframe, ibis_table, core_client):
    t = core_client.from_dataframe(dataframe)
    result = t.execute()
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    t = core_client.from_dataframe(dataframe, name="foo")
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)

    client = core_client
    t = core_client.from_dataframe(dataframe, name="foo", client=client)
    expected = ibis_table.execute()
    tm.assert_frame_equal(result, expected)


def test_array_literal_from_series(core_client):
    values = [1, 2, 3, 4]
    s = dd.from_pandas(pd.Series(values), npartitions=1)
    expr = ibis.array(s)

    assert expr.equals(ibis.array(values))
    assert core_client.execute(expr) == pytest.approx([1, 2, 3, 4])


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
    param = ibis.param("int64")
    result = execute(param.op(), params={param.op(): 42})
    assert result == 42


def test_missing_data_sources():
    t = ibis.table([("a", "string")])
    expr = t.a.length()
    with pytest.raises(com.UnboundExpressionError):
        execute(expr.op())


def test_post_execute_called_on_joins(dataframe, core_client, ibis_table):
    count = [0]

    @post_execute.register(ops.InnerJoin, dd.DataFrame)
    def tmp_left_join_exe(op, lhs, **kwargs):
        count[0] += 1
        return lhs

    left = ibis_table
    right = left.view()
    join = left.join(right, "plain_strings")[left.plain_int64]
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
    assert pandas_execute_node.dispatch(*types).__name__ == "raise_unknown_op"
