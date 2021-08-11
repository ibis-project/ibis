import pandas.testing as tm


def test_isnull(client):
    table = client.table('nan_table')
    table_pandas = table.compile().toPandas()

    for (col, _) in table_pandas.iteritems():
        result = (
            table[table[col].isnull()]
            .compile()
            .toPandas()
            .reset_index(drop=True)
        )
        expected = table_pandas[table_pandas[col].isnull()].reset_index(
            drop=True
        )
        tm.assert_frame_equal(result, expected)


def test_notnull(client):
    table = client.table('nan_table')
    table_pandas = table.compile().toPandas()

    for (col, _) in table_pandas.iteritems():
        result = (
            table[table[col].notnull()]
            .compile()
            .toPandas()
            .reset_index(drop=True)
        )
        expected = table_pandas[table_pandas[col].notnull()].reset_index(
            drop=True
        )
        tm.assert_frame_equal(result, expected)
