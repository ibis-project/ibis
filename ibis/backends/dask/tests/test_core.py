from typing import Any

import pytest
from multipledispatch.conflict import ambiguities

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.pandas.dispatch import execute_node as pandas_execute_node
from ibis.expr.scope import Scope

dd = pytest.importorskip("dask.dataframe")

from dask.dataframe.utils import tm  # noqa: E402

from ibis.backends.dask import Backend  # noqa: E402

from ..core import execute, is_computable_input  # noqa: E402
from ..dispatch import execute_node  # noqa: E402


@pytest.mark.parametrize('func', [execute_node])
def test_no_execute_ambiguities(func):
    assert not ambiguities(func.funcs)


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


def test_execute_parameter_only():
    param = ibis.param('int64')
    result = execute(param, params={param: 42})
    assert result == 42


def test_missing_data_sources():
    t = ibis.table([('a', 'string')])
    expr = t.a.length()
    with pytest.raises(com.UnboundExpressionError):
        execute(expr)


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
        match=(
            'Could not find signature for execute_node: '
            '<DatabaseTable, MyBackend>'
        ),
    ):
        con.execute(t)


def test_is_computable_input():
    class MyObject:
        def __init__(self, value: float) -> None:
            self.value = value

        def __getattr__(self, name: str) -> Any:
            return getattr(self.value, name)

        def __hash__(self) -> int:
            return hash((type(self), self.value))

        def __eq__(self, other):
            return (
                isinstance(other, type(self))
                and isinstance(self, type(other))
                and self.value == other.value
            )

        def __float__(self) -> float:
            return self.value

    @execute_node.register(ops.Add, int, MyObject)
    def add_int_my_object(op, left, right, **kwargs):
        return left + right.value

    # This multimethod must be implemented to play nicely with other value
    # types like columns and literals. In other words, for a custom
    # non-expression object to play nicely it must somehow map to one of the
    # types in ibis/expr/datatypes.py
    @dt.infer.register(MyObject)
    def infer_my_object(_, **kwargs):
        return dt.float64

    @is_computable_input.register(MyObject)
    def is_computable_input_my_object(_):
        return True

    one = ibis.literal(1)
    two = MyObject(2.0)
    assert is_computable_input(two)

    three = one + two
    four = three + 1
    result = execute(four)
    assert result == 4.0

    del execute_node[ops.Add, int, MyObject]

    execute_node.reorder()
    execute_node._cache.clear()

    del dt.infer.funcs[(MyObject,)]
    dt.infer.reorder()
    dt.infer._cache.clear()


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
