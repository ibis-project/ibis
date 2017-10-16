import uuid

import numpy as np
import pytest
import pandas as pd
import pandas.util.testing as tm
import google.datalab.bigquery as bq

import ibis
import ibis.expr.types as ir
from ibis.bigquery.tests import conftest as conftest


def test_table(client_with_table):
    # table must exist
    table = client_with_table.table(conftest.TABLE)
    assert isinstance(table, ir.TableExpr)


def test_array_execute(table, df):
    col_name = df.select_dtypes(['float']).columns[0]
    expr = table[col_name]
    result = expr.execute()[col_name]
    expected = df[col_name]
    tm.assert_series_equal(result, expected)


def test_literal_execute(client):
    expected = '1234'
    expr = ibis.literal(expected)
    result = client.execute(expr).iloc[0, 0]
    assert result == expected


def test_simple_aggregate_execute(table, df):
    col_name = df.select_dtypes(['float']).columns[0]
    expr = table[col_name].sum()
    result = expr.execute()
    expected = df[col_name].sum()
    np.testing.assert_allclose(result, expected)


def test_list_tables(client_with_table):
    assert len(client_with_table.list_tables(like=conftest.TABLE)) == 1


def test_database_layer(client_with_table):
    bq_dataset = client_with_table._proxy.get_dataset(
        client_with_table.dataset_id)
    actual = client_with_table.list_tables()
    expected = [el.name.table_id for el in bq_dataset.tables()]
    assert sorted(actual) == sorted(expected)


def test_compile_verify(table):
    column = table['string_column']
    unsupported_expr = column.replace('foo', 'bar')
    supported_expr = column.lower()
    assert not unsupported_expr.verify()
    assert supported_expr.verify()


def test_compile_toplevel():
    t = ibis.table([('foo', 'double')], name='t0')

    # it works!
    expr = t.foo.sum()
    result = ibis.bigquery.compile(expr)
    # FIXME: remove quotes because bigquery can't use anythig that needs
    # quoting?
    expected = """\
SELECT sum(`foo`) AS `sum`
FROM t0"""  # noqa
    assert str(result) == expected


@pytest.mark.xfail
def test_df_upload(client):
    expected = pd.DataFrame(dict(a=[1], b=[2.], c=['a'], d=[True]))
    schema = bq.Schema.from_data(expected)
    t = client.table('rando', schema)
    t.upload(expected)
    result = t.execute()
    t.delete()
    assert result.equals(expected)
    assert not t.exists()


@pytest.mark.xfail()
def test_create_and_drop_table(client):
    t = client.table(conftest.TABLE)
    name = str(uuid.uuid4())
    client.create_table(name, t.limit(5))
    new_table = client.table(name)
    tm.assert_frame_equal(new_table.execute(), t.limit(5).execute())
    client.drop_table(name)
    assert name not in client.list_tables()
