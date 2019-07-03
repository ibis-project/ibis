import ibis
import ibis.expr.datatypes as dt


def test_list_databases(client):
    dbs = client.list_databases()
    assert dbs == ['default']


def test_list_tables(client):
    tables = client.list_tables()
    assert tables == ['nested_types', 'simple', 'struct']


def test_get_schema(client):
    schema = client.get_schema('simple')
    assert schema == ibis.schema([('foo', 'int64'),
                                  ('bar', 'string')])


def test_table_simple(client):
    db = client.database()
    # also tests that client.table and db.table give the same result
    for t in [client.table('simple'), db.table('simple')]:
        assert t.columns == ['foo', 'bar']
        result = t.execute()
        assert len(result) == 1
        assert list(result.iloc[0]) == [1, 'a']


def test_struct_type(struct):
    assert struct.columns == ['struct_col']

    t = struct.schema().types
    assert isinstance(t[0], dt.Struct)
    assert t[0].names == ['_1', '_2', '_3']
    assert isinstance(t[0].types[0], dt.Int64)
    assert isinstance(t[0].types[1], dt.Int64)
    assert isinstance(t[0].types[2], dt.String)


def test_table_nested_types(nested_types):
    assert nested_types.columns == [
        'list_of_ints',
        'list_of_list_of_ints',
        'map_tuple_list_of_list_of_ints'
    ]

    t = nested_types.schema().types
    assert isinstance(t[0], dt.Array)
    assert isinstance(t[0].value_type, dt.Int64)
    assert isinstance(t[1], dt.Array)
    assert isinstance(t[1].value_type, dt.Array)
    assert isinstance(t[1].value_type.value_type, dt.Int64)
    assert isinstance(t[2], dt.Map)
    assert isinstance(t[2].key_type, dt.Struct)
    assert t[2].key_type.names == ['_1', '_2']
    assert isinstance(t[2].key_type.types[0], dt.Int64)
    assert isinstance(t[2].key_type.types[1], dt.Int64)
    assert isinstance(t[2].value_type, dt.Array)
    assert isinstance(t[2].value_type.value_type, dt.Array)
    assert isinstance(t[2].value_type.value_type.value_type, dt.Int64)


def test_current_database(client):
    assert client.current_database == 'default'
