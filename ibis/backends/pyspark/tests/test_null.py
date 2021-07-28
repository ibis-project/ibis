import pandas.testing as tm
import pytest

pytestmark = pytest.mark.pyspark


@pytest.mark.parametrize(
    'is_null_fn', [lambda t, c: t[c].isnull(), lambda t, c: t[c].notnull()]
)
def test_isnull(client, is_null_fn):
    table = client.table('nan_table')
    table_pandas = table.compile().toPandas()

    for (col, _) in table_pandas.iteritems():
        ibis_is_null_expr = is_null_fn(table, col)
        result = (
            table[ibis_is_null_expr]
            .compile()
            .toPandas()
            .reset_index(drop=True)
        )
        pandas_is_null_series = is_null_fn(table_pandas, col)
        expected = table_pandas[pandas_is_null_series].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)


def test_isna(client):
    table = client.table('nan_table').select(['age', 'height'])
    table_pandas = table.compile().toPandas()

    for (col, _) in table_pandas.iteritems():
        result = (
            table[table[col].isnan()]
            .compile()
            .toPandas()
            .reset_index(drop=True)
        )

        expected = table_pandas[table_pandas[col].isna()].reset_index(
            drop=True
        )
        tm.assert_frame_equal(result, expected)


def test_fillna(client):
    table = client.table('nan_table').select(['age', 'height'])
    table_pandas = table.compile().toPandas()

    for (col, _) in table_pandas.iteritems():
        result = (
            table.mutate(filled=table[col].fillna(0.0))
            .compile()
            .toPandas()
            .reset_index(drop=True)
        )

        expected = table_pandas.assign(
            filled=table_pandas[col].fillna(0.0)
        ).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)
