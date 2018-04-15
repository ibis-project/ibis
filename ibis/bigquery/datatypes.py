import six

from multipledispatch import Dispatcher

import ibis.expr.datatypes as dt


class TypeTranslationContext(object):
    """A tag class to allow alteration of the way a particular type is
    translated.

    Notes
    -----
    This is used to translate INT64 types to FLOAT64 when INT64 is used in the
    definition of a UDF.
    """
    __slots__ = ()


ibis_type_to_bigquery_type = Dispatcher('ibis_type_to_bigquery_type')


@ibis_type_to_bigquery_type.register(dt.DataType)
def trans_default(t):
    return ibis_type_to_bigquery_type(t, TypeTranslationContext())


@ibis_type_to_bigquery_type.register(six.string_types)
def trans_default_from_string(string):
    return ibis_type_to_bigquery_type(dt.dtype(string))


@ibis_type_to_bigquery_type.register(dt.Floating, TypeTranslationContext)
def trans_float64(t, context):
    return 'FLOAT64'


@ibis_type_to_bigquery_type.register(dt.Integer, TypeTranslationContext)
def trans_integer(t, context):
    return 'INT64'


@ibis_type_to_bigquery_type.register(dt.Array, TypeTranslationContext)
def trans_array(t, context):
    return 'ARRAY<{}>'.format(
        ibis_type_to_bigquery_type(t.value_type, context))


@ibis_type_to_bigquery_type.register(dt.Struct, TypeTranslationContext)
def trans_struct(t, context):
    return 'STRUCT<{}>'.format(
        ', '.join(
            '{} {}'.format(name, ibis_type_to_bigquery_type(type, context))
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


@ibis_type_to_bigquery_type.register(six.string_types, TypeTranslationContext)
def trans_str(t, context):
    return ibis_type_to_bigquery_type(dt.dtype(t), context)
