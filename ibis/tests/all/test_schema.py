import ibis


def test_types(backend, alltypes, df):
    expected = ibis.schema([
        ('index', 'int64'),
        ('Unnamed: 0', 'int64'),
        ('id', 'int32'),
        ('bool_col', 'boolean'),
        ('tinyint_col', 'int8'),
        ('smallint_col', 'int16'),
        ('int_col', 'int32'),
        ('bigint_col', 'int64'),
        ('float_col', 'float'),
        ('double_col', 'double'),
        ('date_string_col', 'string'),
        ('string_col', 'string'),
        ('timestamp_col', 'timestamp'),
        ('year', 'int32'),
        ('month', 'int32')
    ])

    print(alltypes.schema())

    assert alltypes.schema() == expected

