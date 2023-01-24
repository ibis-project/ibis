from __future__ import annotations

from multipledispatch import Dispatcher

import ibis.expr.datatypes as dt

ibis_type_to_bigquery_type = Dispatcher("ibis_type_to_bigquery_type")


@ibis_type_to_bigquery_type.register(str)
def trans_string_default(datatype):
    return ibis_type_to_bigquery_type(dt.dtype(datatype))


@ibis_type_to_bigquery_type.register(dt.Floating)
def trans_float64(t):
    return "FLOAT64"


@ibis_type_to_bigquery_type.register(dt.Integer)
def trans_integer(t):
    return "INT64"


@ibis_type_to_bigquery_type.register(dt.Binary)
def trans_binary(t):
    return "BYTES"


@ibis_type_to_bigquery_type.register(dt.UInt64)
def trans_lossy_integer(t):
    raise TypeError("Conversion from uint64 to BigQuery integer type (int64) is lossy")


@ibis_type_to_bigquery_type.register(dt.Array)
def trans_array(t):
    return f"ARRAY<{ibis_type_to_bigquery_type(t.value_type)}>"


@ibis_type_to_bigquery_type.register(dt.Struct)
def trans_struct(t):
    return "STRUCT<{}>".format(
        ", ".join(
            f"{name} {ibis_type_to_bigquery_type(dt.dtype(type_))}"
            for name, type_ in t.fields.items()
        )
    )


@ibis_type_to_bigquery_type.register(dt.Date)
def trans_date(t):
    return "DATE"


@ibis_type_to_bigquery_type.register(dt.Timestamp)
def trans_timestamp(t):
    if t.timezone is not None:
        raise TypeError("BigQuery does not support timestamps with timezones")
    return "TIMESTAMP"


@ibis_type_to_bigquery_type.register(dt.DataType)
def trans_type(t):
    return str(t).upper()


@ibis_type_to_bigquery_type.register(dt.Decimal)
def trans_numeric(t):
    if (t.precision, t.scale) != (38, 9):
        raise TypeError(
            "BigQuery only supports decimal types with precision of 38 and "
            "scale of 9"
        )
    return "NUMERIC"


@ibis_type_to_bigquery_type.register(dt.JSON)
def trans_json(t):
    return "JSON"


def spread_type(dt: dt.DataType):
    """Returns a generator that contains all the types in the given type.

    For complex types like set and array, it returns the types of the elements.
    """
    if dt.is_array():
        yield from spread_type(dt.value_type)
    elif dt.is_struct():
        for type_ in dt.types:
            yield from spread_type(type_)
    elif dt.is_map():
        yield from spread_type(dt.key_type)
        yield from spread_type(dt.value_type)
    yield dt
