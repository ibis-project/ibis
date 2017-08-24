

_sql_type_names = {
    'int8': 'Int8',
    'int16': 'Int16',
    'int32': 'Int32',
    'int64': 'Int64',
    'float': 'Float32',
    'double': 'Float64',
    'string': 'String',
    'boolean': 'UInt8',
    'date': 'Date',
    'timestamp': 'DateTime',
    'decimal': 'UInt64',  # see Clickhouse issue #253
}


def _type_to_sql_string(tval):
    return _sql_type_names[tval.name.lower()]
