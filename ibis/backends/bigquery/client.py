"""BigQuery ibis client implementation."""

from __future__ import annotations

import functools

import google.cloud.bigquery as bq
import pandas as pd

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import Database
from ibis.backends.bigquery.datatypes import ibis_type_to_bigquery_type

NATIVE_PARTITION_COL = "_PARTITIONTIME"


_DTYPE_TO_IBIS_TYPE = {
    "INT64": dt.int64,
    "FLOAT64": dt.double,
    "BOOL": dt.boolean,
    "STRING": dt.string,
    "DATE": dt.date,
    # FIXME: enforce no tz info
    "DATETIME": dt.timestamp,
    "TIME": dt.time,
    "TIMESTAMP": dt.timestamp,
    "BYTES": dt.binary,
    "NUMERIC": dt.Decimal(38, 9),
}


_LEGACY_TO_STANDARD = {
    "INTEGER": "INT64",
    "FLOAT": "FLOAT64",
    "BOOLEAN": "BOOL",
}


@dt.dtype.register(bq.schema.SchemaField)
def bigquery_field_to_ibis_dtype(field):
    """Convert BigQuery `field` to an ibis type."""
    typ = field.field_type
    if typ == "RECORD":
        fields = field.fields
        assert fields, "RECORD fields are empty"
        names = [el.name for el in fields]
        ibis_types = list(map(dt.dtype, fields))
        ibis_type = dt.Struct(dict(zip(names, ibis_types)))
    else:
        ibis_type = _LEGACY_TO_STANDARD.get(typ, typ)
        ibis_type = _DTYPE_TO_IBIS_TYPE.get(ibis_type, ibis_type)
    if field.mode == "REPEATED":
        ibis_type = dt.Array(ibis_type)
    return ibis_type


@sch.infer.register(bq.table.Table)
def bigquery_schema(table):
    """Infer the schema of a BigQuery `table` object."""
    fields = {el.name: dt.dtype(el) for el in table.schema}
    partition_info = table._properties.get("timePartitioning", None)

    # We have a partitioned table
    if partition_info is not None:
        partition_field = partition_info.get("field", NATIVE_PARTITION_COL)

        # Only add a new column if it's not already a column in the schema
        fields.setdefault(partition_field, dt.timestamp)
    return sch.schema(fields)


class BigQueryCursor:
    """BigQuery cursor.

    This allows the BigQuery client to reuse machinery in
    :file:`ibis/client.py`.
    """

    def __init__(self, query):
        """Construct a BigQueryCursor with query `query`."""
        self.query = query

    def fetchall(self):
        """Fetch all rows."""
        result = self.query.result()
        return [row.values() for row in result]

    @property
    def columns(self):
        """Return the columns of the result set."""
        result = self.query.result()
        return [field.name for field in result.schema]

    @property
    def description(self):
        """Get the fields of the result set's schema."""
        result = self.query.result()
        return list(result.schema)

    def __enter__(self):
        """No-op for compatibility."""
        return self

    def __exit__(self, *_):
        """No-op for compatibility."""


class BigQueryDatabase(Database):
    """A BigQuery dataset."""


@functools.singledispatch
def bigquery_param(dtype, value, name):
    raise NotADirectoryError(dtype)


@bigquery_param.register
def bq_param_struct(dtype: dt.Struct, value, name):
    fields = dtype.fields
    field_params = [bigquery_param(fields[k], v, k) for k, v in value.items()]
    result = bq.StructQueryParameter(name, *field_params)
    return result


@bigquery_param.register
def bq_param_array(dtype: dt.Array, value, name):
    value_type = dtype.value_type

    try:
        bigquery_type = ibis_type_to_bigquery_type(value_type)
    except NotImplementedError:
        raise com.UnsupportedBackendType(dtype)
    else:
        if isinstance(value_type, dt.Struct):
            query_value = [
                bigquery_param(dtype.value_type, struct, f"element_{i:d}")
                for i, struct in enumerate(value)
            ]
            bigquery_type = "STRUCT"
        elif isinstance(value_type, dt.Array):
            raise TypeError("ARRAY<ARRAY<T>> is not supported in BigQuery")
        else:
            query_value = value
        result = bq.ArrayQueryParameter(name, bigquery_type, query_value)
        return result


@bigquery_param.register
def bq_param_timestamp(_: dt.Timestamp, value, name):
    # TODO(phillipc): Not sure if this is the correct way to do this.
    timestamp_value = pd.Timestamp(value, tz="UTC").to_pydatetime()
    return bq.ScalarQueryParameter(name, "TIMESTAMP", timestamp_value)


@bigquery_param.register
def bq_param_string(_: dt.String, value, name):
    return bq.ScalarQueryParameter(name, "STRING", value)


@bigquery_param.register
def bq_param_integer(_: dt.Integer, value, name):
    return bq.ScalarQueryParameter(name, "INT64", value)


@bigquery_param.register
def bq_param_double(_: dt.Floating, value, name):
    return bq.ScalarQueryParameter(name, "FLOAT64", value)


@bigquery_param.register
def bq_param_boolean(_: dt.Boolean, value, name):
    return bq.ScalarQueryParameter(name, "BOOL", value)


@bigquery_param.register
def bq_param_date(_: dt.Date, value, name):
    return bq.ScalarQueryParameter(
        name, "DATE", pd.Timestamp(value).to_pydatetime().date()
    )


class BigQueryTable(ops.DatabaseTable):
    pass


def rename_partitioned_column(table_expr, bq_table, partition_col):
    """Rename native partition column to user-defined name."""
    partition_info = bq_table._properties.get("timePartitioning", None)

    # If we don't have any partiton information, the table isn't partitioned
    if partition_info is None:
        return table_expr

    # If we have a partition, but no "field" field in the table properties,
    # then use NATIVE_PARTITION_COL as the default
    partition_field = partition_info.get("field", NATIVE_PARTITION_COL)

    # The partition field must be in table_expr columns
    assert partition_field in table_expr.columns

    # No renaming if the config option is set to None or the partition field
    # is not _PARTITIONTIME
    if partition_col is None or partition_field != NATIVE_PARTITION_COL:
        return table_expr
    return table_expr.relabel({NATIVE_PARTITION_COL: partition_col})


def parse_project_and_dataset(project: str, dataset: str = "") -> tuple[str, str, str]:
    """Compute the billing project, data project, and dataset if available.

    This function figure out the project id under which queries will run versus
    the project of where the data live as well as what dataset to use.

    Parameters
    ----------
    project : str
        A project name
    dataset : Optional[str]
        A ``<project>.<dataset>`` string or just a dataset name

    Examples
    --------
    >>> data_project, billing_project, dataset = parse_project_and_dataset(
    ...     'ibis-gbq',
    ...     'foo-bar.my_dataset'
    ... )
    >>> data_project
    'foo-bar'
    >>> billing_project
    'ibis-gbq'
    >>> dataset
    'my_dataset'
    >>> data_project, billing_project, dataset = parse_project_and_dataset(
    ...     'ibis-gbq',
    ...     'my_dataset'
    ... )
    >>> data_project
    'ibis-gbq'
    >>> billing_project
    'ibis-gbq'
    >>> dataset
    'my_dataset'
    >>> data_project, billing_project, dataset = parse_project_and_dataset(
    ...     'ibis-gbq'
    ... )
    >>> data_project
    'ibis-gbq'
    >>> print(dataset)
    None
    """
    if dataset.count(".") > 1:
        raise ValueError(
            "{} is not a BigQuery dataset. More info https://cloud.google.com/bigquery/docs/datasets-intro".format(
                dataset
            )
        )
    elif dataset.count(".") == 1:
        data_project, dataset = dataset.split(".")
        billing_project = project
    else:
        billing_project = data_project = project

    return data_project, billing_project, dataset
