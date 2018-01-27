import pytest

import pandas as pd
import pandas.util.testing as tm

import ibis

pytest.importorskip('multipledispatch')

from ibis.pandas.client import PandasTable  # noqa: E402

pytestmark = pytest.mark.pandas


@pytest.fixture
def client():
    return ibis.pandas.connect({
        'df': pd.DataFrame({'a': [1, 2, 3], 'b': list('abc')}),
        'df_unknown': pd.DataFrame({
            'array_of_strings': [['a', 'b'], [], ['c']],
        }),
    })


@pytest.fixture
def table(client):
    return client.table('df')


def test_client_table(table):
    assert isinstance(table.op(), ibis.expr.operations.DatabaseTable)
    assert isinstance(table.op(), PandasTable)


def test_client_table_repr(table):
    assert 'PandasTable' in repr(table)


def test_literal(client):
    assert client.execute(ibis.literal(1)) == 1


def test_read_with_undiscoverable_type(client):
    with pytest.raises(TypeError):
        client.table('df_unknown')


def test_drop(table):
    table = table.mutate(c=table.a)
    expr = table.drop(['a'])
    result = expr.execute()
    expected = table[['b', 'c']].execute()
    tm.assert_frame_equal(result, expected)
