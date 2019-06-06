import ibis


def test_list_databases(client_empty):
    dbs = client_empty.list_databases()
    assert dbs == ['default']


def test_list_tables(client_simple):
    tables = client_simple.list_tables()
    assert tables == ['t']


def test_get_schema(client_simple):
    schema = client_simple.get_schema('t')
    assert schema == ibis.schema([('foo', 'int64'),
                                  ('bar', 'string')])


def test_current_database(client_simple):
    assert client_simple.current_database == 'default'
