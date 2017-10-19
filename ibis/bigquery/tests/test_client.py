import pytest

import numpy as np
import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.types as ir


pytestmark = pytest.mark.bigquery
pytest.importorskip('google.cloud.bigquery')


def test_table(alltypes):
    assert isinstance(alltypes, ir.TableExpr)


def test_array_execute(alltypes, df):
    col_name = 'float_col'
    expr = alltypes[col_name]
    result = expr.execute()[col_name]
    expected = df[col_name]
    tm.assert_series_equal(result, expected)


def test_literal_execute(client):
    expected = '1234'
    expr = ibis.literal(expected)
    result = client.execute(expr).iloc[0, 0]
    assert result == expected


def test_simple_aggregate_execute(table, df):
    col_name = 'float_col'
    expr = table[col_name].sum()
    result = expr.execute()
    expected = df[col_name].sum()
    np.testing.assert_allclose(result, expected)


def test_list_tables(client, table_id):
    assert len(client.list_tables(like=table_id)) == 1


def test_database_layer(client):
    bq_dataset = client._proxy.get_dataset(client.dataset_id)
    actual = client.list_tables()
    expected = [el.name for el in bq_dataset.list_tables()]
    assert sorted(actual) == sorted(expected)


def test_compile_verify(alltypes):
    column = alltypes['string_col']
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
