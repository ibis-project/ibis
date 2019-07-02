import ibis


def test_list_databases(client):
    dbs = client.list_databases()
    assert dbs == ['default']


def test_list_tables(client):
    tables = client.list_tables()
    assert tables == ['nested_types', 'simple']


def test_get_schema(client):
    schema = client.get_schema('simple')
    assert schema == ibis.schema([('foo', 'int64'),
                                  ('bar', 'string')])


def test_table_simple(client):
    db = client.database()
    correct_string = ''.join([
        'SparkTable[table]\n  ',
        'name: simple\n  schema:\n    ',
        'foo : int64\n    bar : string'
    ])
    assert str(client.table('simple')) == correct_string
    assert str(db.table('simple')) == correct_string


def test_table_nested_types(nested_types):
    correct_string = ''.join([
        'SparkTable[table]\n  ',
        'name: nested_types\n  schema:\n    ',
        'c1 : array<int64>\n    c2 : array<array<int64>>\n    ',
        'c3 : map<struct<_1: None, _2: None>, array<array<int64>>>'
    ])
    assert str(nested_types) == correct_string


def test_current_database(client):
    assert client.current_database == 'default'
