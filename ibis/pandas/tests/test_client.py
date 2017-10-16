import pytest

import pandas as pd

import ibis

pytest.importorskip('multipledispatch')

from ibis.pandas.client import PandasTable  # noqa: E402

pytestmark = pytest.mark.pandas


@pytest.fixture
def client():
    return ibis.pandas.connect({
        'df': pd.DataFrame({'a': [1, 2, 3]}),
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
