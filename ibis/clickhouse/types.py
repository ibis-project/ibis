from toolz import itemmap, merge, pluck

# TODO cleanup, reorganize

_sql_type_names = {
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
    'decimal': 'UInt64',  # see Clickhouse issue #253
}


def _type_to_sql_string(tval):
    return _sql_type_names[tval.name.lower()]


# Nullable(UInt8) etc.
_dtypes = {'object': 'String',
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
           'datetime64[D]': 'Date',
           'datetime64[ns]': 'DateTime'}

PD2CH = merge(_dtypes, {'bool': 'UInt8'})
CH2PD = merge(itemmap(reversed, _dtypes), {'Null': 'object'})


# what is the correct signature of an unsigned ibis type?
# TODO: set, enum, array
CH2IB = {'UInt64': 'int64',
         'UInt32': 'int32',
         'UInt16': 'int16',
         'UInt8': 'int8',
         'Int64': 'int64',
         'Int32': 'int32',
         'Int16': 'int16',
         'Int8': 'int8',
         'Float64': 'float',
         'Float32': 'float',
         'String': 'string',
         'FixedString': 'string',
         'Date': 'date',
         'DateTime': 'timestamp'}
