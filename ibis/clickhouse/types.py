pandas_to_clickhouse = {
    'object': 'String',
    'uint64': 'UInt64',
    'uint32': 'UInt32',
    'uint16': 'UInt16',
    'float64': 'Float64',
    'float32': 'Float32',
    'uint8': 'UInt8',
    'int64': 'Int64',
    'int32': 'Int32',
    'int16': 'Int16',
    'int8': 'Int8',
    'bool': 'UInt8',
    'datetime64[D]': 'Date',
    'datetime64[ns]': 'DateTime'
}

clickhouse_to_pandas = {
    'UInt8': 'uint8',
    'UInt16': 'uint16',
    'UInt32': 'uint32',
    'UInt64': 'uint64',
    'Int8': 'int8',
    'Int16': 'int16',
    'Int32': 'int32',
    'Int64': 'int64',
    'Float64': 'float64',
    'Float32': 'float32',
    'String': 'object',
    'FixedString': 'object',  # TODO
    'Null': 'object',
    'Date': 'datetime64[D]',
    'DateTime': 'datetime64[ns]',
    'Nullable(UInt8)': 'float32',
    'Nullable(UInt16)': 'float32',
    'Nullable(UInt32)': 'float32',
    'Nullable(UInt64)': 'float64',
    'Nullable(Int8)': 'float32',
    'Nullable(Int16)': 'float32',
    'Nullable(Int32)': 'float32',
    'Nullable(Int64)': 'float64',
    'Nullable(Float32)': 'float32',
    'Nullable(Float64)': 'float64',
    'Nullable(String)': 'object',
    'Nullable(FixedString)': 'object',  # TODO
    'Nullable(Date)': 'Date',
    'Nullable(DateTime)': 'DateTime'
}

ibis_to_clickhouse = {
    'null': 'Null',
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
    'decimal': 'UInt64'  # see yandex/clickhouse#253
}

clickhouse_to_ibis = {
    'Null': 'null',
    'UInt64': 'int64',
    'UInt32': 'int32',
    'UInt16': 'int16',
    'UInt8': 'int8',
    'Int64': 'int64',
    'Int32': 'int32',
    'Int16': 'int16',
    'Int8': 'int8',
    'Float64': 'double',
    'Float32': 'float',
    'String': 'string',
    'FixedString': 'string',
    'Date': 'date',
    'DateTime': 'timestamp'
}
