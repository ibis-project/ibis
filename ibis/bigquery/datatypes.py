from multipledispatch import Dispatcher

import ibis.expr.datatypes as dt


class TypeTranslationContext:
    """A tag class to allow alteration of the way a particular type is
    translated.

    Notes
    -----
    This is used to raise an exception when INT64 types are encountered to
    avoid suprising results due to BigQuery's handling of INT64 types in
    JavaScript UDFs.
    """

    __slots__ = ()


class UDFContext(TypeTranslationContext):
    __slots__ = ()


ibis_type_to_bigquery_type = Dispatcher('ibis_type_to_bigquery_type')


@ibis_type_to_bigquery_type.register(str)
def trans_string_default(datatype):
    return ibis_type_to_bigquery_type(dt.dtype(datatype))


@ibis_type_to_bigquery_type.register(dt.DataType)
def trans_default(t):
    return ibis_type_to_bigquery_type(t, TypeTranslationContext())


@ibis_type_to_bigquery_type.register(str, TypeTranslationContext)
def trans_string_context(datatype, context):
    return ibis_type_to_bigquery_type(dt.dtype(datatype), context)


@ibis_type_to_bigquery_type.register(dt.Floating, TypeTranslationContext)
def trans_float64(t, context):
    return 'FLOAT64'


@ibis_type_to_bigquery_type.register(dt.Integer, TypeTranslationContext)
def trans_integer(t, context):
    return 'INT64'


@ibis_type_to_bigquery_type.register(
    dt.UInt64, (TypeTranslationContext, UDFContext)
)
def trans_lossy_integer(t, context):
    raise TypeError(
        'Conversion from uint64 to BigQuery integer type (int64) is lossy'
    )


@ibis_type_to_bigquery_type.register(dt.Array, TypeTranslationContext)
def trans_array(t, context):
    return 'ARRAY<{}>'.format(
        ibis_type_to_bigquery_type(t.value_type, context)
    )


@ibis_type_to_bigquery_type.register(dt.Struct, TypeTranslationContext)
def trans_struct(t, context):
    return 'STRUCT<{}>'.format(
        ', '.join(
            '{} {}'.format(
                name, ibis_type_to_bigquery_type(dt.dtype(type), context)
            )
            for name, type in zip(t.names, t.types)
        )
    )


@ibis_type_to_bigquery_type.register(dt.Date, TypeTranslationContext)
def trans_date(t, context):
    return 'DATE'


@ibis_type_to_bigquery_type.register(dt.Timestamp, TypeTranslationContext)
def trans_timestamp(t, context):
    if t.timezone is not None:
        raise TypeError('BigQuery does not support timestamps with timezones')
    return 'TIMESTAMP'


@ibis_type_to_bigquery_type.register(dt.DataType, TypeTranslationContext)
def trans_type(t, context):
    return str(t).upper()


@ibis_type_to_bigquery_type.register(dt.Integer, UDFContext)
def trans_integer_udf(t, context):
    # JavaScript does not have integers, only a Number class. BigQuery doesn't
    # behave as expected with INT64 inputs or outputs
    raise TypeError(
        'BigQuery does not support INT64 as an argument type or a return type '
        'for UDFs. Replace INT64 with FLOAT64 in your UDF signature and '
        'cast all INT64 inputs to FLOAT64.'
    )


@ibis_type_to_bigquery_type.register(dt.Decimal, TypeTranslationContext)
def trans_numeric(t, context):
    if (t.precision, t.scale) != (38, 9):
        raise TypeError(
            'BigQuery only supports decimal types with precision of 38 and '
            'scale of 9'
        )
    return 'NUMERIC'


@ibis_type_to_bigquery_type.register(dt.Decimal, TypeTranslationContext)
def trans_numeric_udf(t, context):
    raise TypeError('Decimal types are not supported in BigQuery UDFs')
