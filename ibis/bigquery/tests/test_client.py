import pytest

import numpy as np
import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir


pytestmark = pytest.mark.bigquery
pytest.importorskip('google.cloud.bigquery')


def test_table(alltypes):
    assert isinstance(alltypes, ir.TableExpr)


def test_column_execute(alltypes, df):
    col_name = 'float_col'
    expr = alltypes[col_name]
    result = expr.execute()
    expected = df[col_name]
    tm.assert_series_equal(result, expected)


def test_literal_execute(client):
    expected = '1234'
    expr = ibis.literal(expected)
    result = client.execute(expr)
    assert result == expected


def test_simple_aggregate_execute(alltypes, df):
    col_name = 'float_col'
    expr = alltypes[col_name].sum()
    result = expr.execute()
    expected = df[col_name].sum()
    np.testing.assert_allclose(result, expected)


def test_list_tables(client):
    assert len(client.list_tables(like='functional_alltypes')) == 1


def test_current_database(client):
    assert client.current_database.name == 'testing'
    assert client.current_database.name == client.dataset_id
    assert client.current_database.tables == client.list_tables()


def test_database(client):
    database = client.database(client.dataset_id)
    assert database.list_tables() == client.list_tables()


def test_database_layer(client):
    bq_dataset = client._proxy.get_dataset(client.dataset_id)
    actual = client.list_tables()
    expected = [el.name for el in bq_dataset.list_tables()]
    assert sorted(actual) == sorted(expected)


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


def test_struct_field_access(struct_table):
    expr = struct_table.struct_col.string_field
    result = expr.execute()
    expected = pd.Series([None, 'a'], name='tmp')
    tm.assert_series_equal(result, expected)


def test_array_index(struct_table):
    expr = struct_table.array_of_structs_col[1]
    result = expr.execute()
    expected = pd.Series(
        [
            {'int_field': None, 'string_field': None},
            {'int_field': None, 'string_field': 'hijklmnop'}
        ],
        name='tmp'
    )
    tm.assert_series_equal(result, expected)


def test_array_concat(struct_table):
    c = struct_table.array_of_structs_col
    expr = c + c
    result = expr.execute()
    expected = pd.Series(
        [
            [
                {'int_field': 12345, 'string_field': 'abcdefg'},
                {'int_field': None, 'string_field': None},
                {'int_field': 12345, 'string_field': 'abcdefg'},
                {'int_field': None, 'string_field': None},
            ],
            [
                {'int_field': 12345, 'string_field': 'abcdefg'},
                {'int_field': None, 'string_field': 'hijklmnop'},
                {'int_field': 12345, 'string_field': 'abcdefg'},
                {'int_field': None, 'string_field': 'hijklmnop'},
            ],
        ],
        name='tmp',
    )
    tm.assert_series_equal(result, expected)


def test_array_length(struct_table):
    expr = struct_table.array_of_structs_col.length()
    result = expr.execute()
    expected = pd.Series([2, 2], name='tmp')
    tm.assert_series_equal(result, expected)


def test_array_collect(struct_table):
    key = struct_table.array_of_structs_col[0].string_field
    expr = struct_table.groupby(key=key).aggregate(
        foo=lambda t: t.array_of_structs_col[0].int_field.collect()
    )
    result = expr.execute()
    expected = struct_table.execute()
    expected = expected.assign(
        key=expected.array_of_structs_col.apply(lambda x: x[0]['string_field'])
    ).groupby('key').apply(
        lambda df: list(
            df.array_of_structs_col.apply(lambda x: x[0]['int_field'])
        )
    ).reset_index().rename(columns={0: 'foo'})
    tm.assert_frame_equal(result, expected)


def test_count_distinct_with_filter(alltypes):
    expr = alltypes.string_col.nunique(
        where=alltypes.string_col.cast('int64') > 1
    )
    result = expr.execute()
    expected = alltypes.string_col.execute()
    expected = expected[expected.astype('int64') > 1].nunique()
    assert result == expected


@pytest.mark.parametrize('type', ['date', dt.date])
def test_cast_string_to_date(alltypes, df, type):
    import toolz

    string_col = alltypes.date_string_col
    month, day, year = toolz.take(3, string_col.split('/'))

    expr = '20' + ibis.literal('-').join([year, month, day])
    expr = expr.cast(type)
    result = expr.execute().astype(
        'datetime64[ns]'
    ).sort_values().reset_index(drop=True).rename('date_string_col')
    expected = pd.to_datetime(
        df.date_string_col
    ).dt.normalize().sort_values().reset_index(drop=True)
    tm.assert_series_equal(result, expected)


def test_has_partitions(alltypes, parted_alltypes, client):
    col = ibis.options.bigquery.partition_col
    assert col not in alltypes.columns
    assert col in parted_alltypes.columns


def test_different_partition_col_name(client):
    col = ibis.options.bigquery.partition_col = 'FOO_BAR'
    alltypes = client.table('functional_alltypes')
    parted_alltypes = client.table('functional_alltypes_parted')
    assert col not in alltypes.columns
    assert col in parted_alltypes.columns
