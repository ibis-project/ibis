import pandas.util.testing as tm


def test_list_tables(client):
    actual = [t.name for t in client.list_tables()]
    assert ['table1', 'table2'] == actual


def test_execute_table(client, df1):
    expr = client.table('table1')
    result = client.execute(expr)

    tm.assert_frame_equal(
        result.sort_values(by='key').reset_index(drop=True),
        df1.sort_values(by='key').reset_index(drop=True)
    )


def test_execute_join(client, df1, df2):
    t1, t2 = client.table('table1'), client.table('table2')
    joined = t1.join(t2, 'key')
    expr = joined[t1, t2.val2]
    result = client.execute(expr)

    tm.assert_frame_equal(
        result.sort_values(by='key').reset_index(drop=True),
        df1.merge(df2, on='key').sort_values(by='key').reset_index(drop=True)
    )
