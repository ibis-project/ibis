import re

import numpy as np
import pandas as pd

import ibis
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
from ibis.client import (Database, SQLClient)
from ibis.bigquery import compiler as comp
import google.cloud.bigquery


def _ensure_split(table_id, dataset_id):
    split = table_id.split('.')
    if len(split) > 1:
        assert len(split) == 2
        if dataset_id:
            raise ValueError(
                "Can't pass a fully qualified table name *AND* a dataset_id"
            )
        (dataset_id, table_id) = split
    return (table_id, dataset_id)


class _BigQueryAPIProxy():

    def __init__(self, project_id):
        self._client = google.cloud.bigquery.Client(project_id)

    @property
    def client(self):
        return self._client

    @property
    def project_id(self):
        return self.client.project

    def get_datasets(self):
        return list(self.client.list_datasets())

    def get_dataset(self, dataset_id):
        # no need to .reload() to get info?
        return self.client.dataset(dataset_id)

    def get_table(self, table_id, dataset_id, *, reload=True):
        (table_id, dataset_id) = _ensure_split(table_id, dataset_id)
        table = self.client.dataset(dataset_id).table(table_id)
        if reload:
            table.reload()
        return table

    def get_schema(self, table_id, dataset_id):
        return self.get_table(table_id, dataset_id).schema


class BigQueryDataset(Database):
    pass


class BigQueryClient(SQLClient):

    database_class = BigQueryDataset

    def __init__(self, project_id, dataset_id):
        self._proxy = _BigQueryAPIProxy(project_id)
        self._dataset_id = dataset_id

    @property
    def project_id(self):
        return self._proxy.project_id

    @property
    def dataset_id(self):
        return self._dataset_id

    def _build_ast(self, expr, params=None):
        return comp.build_ast(expr, params=params)

    def execute(self, expr, limit='default', async=False, params=None,
                output_options=None):
        if limit != 'default' or async or params:
            raise NotImplementedError()

        stmt = ibis.bigquery.compile(expr)
        query = google.cloud.bigquery.query.QueryResults(
            stmt.replace('`', ''),
            self._proxy.client,
        )
        query.run()
        columns = [el.name for el in query.schema]
        df = pd.DataFrame(list(query.fetch_data()), columns=columns)
        return df

    @property
    def _table_expr_klass(self):
        return ir.TableExpr

    def _fully_qualified_name(self, table_id, dataset_id=None):
        dataset_id = dataset_id or self.dataset_id
        return dataset_id + '.' + table_id

    def _get_table_schema(self, qualified_name):
        return self.get_schema(qualified_name)

    def list_tables(self, like=None, dataset=None):
        dataset = self._proxy.get_dataset(dataset or self.dataset_id)
        result = [table.name for table in dataset.list_tables()]
        if like:
            result = [table_name
                      for table_name in result if re.match(like, table_name)]
        return result

    def set_dataset(self, name):
        self._dataset_id = name

    def exists_dataset(self, name):
        return self._proxy.get_dataset(name).exists()

    def list_datasets(self, like=None):
        results = [dataset.name
                   for dataset in self._proxy.get_datasets()]
        if like:
            results = [dataset_name
                       for dataset_name in results
                       if re.match(like, dataset_name)
                       ]
        return results

    def exists_table(self, name, dataset=None):
        (table_id, dataset_id) = _ensure_split(name, dataset)
        return self._proxy.get_table(table_id, dataset_id).exists()

    def get_schema(self, name, dataset=None):
        (table_id, dataset_id) = _ensure_split(name, dataset)
        bq_table = self._proxy.get_table(table_id, dataset_id)
        return bigquery_dtypes_to_ibis_schema(bq_table)


_DTYPE_TO_IBIS_TYPE = {
    'INT64': dt.int64,
    'FLOAT64': dt.double,
    'BOOL': dt.boolean,
    'STRING': dt.string,
    'DATE': dt.date,
    # FIXME: enforce no tz info
    'DATETIME': dt.timestamp,
    'TIME': dt.time,
    'TIMESTAMP': dt.timestamp,
    # 'BYTES': None,
    # 'ARRAY': None,
    # 'STRUCT': None,
}


_LEGACY_TO_STANDARD = {
    'INTEGER': 'INT64',
    'FLOAT': 'FLOAT64',
    'BOOLEAN': 'BOOL',
}


def _discover_type(field):
    typ = field.field_type
    if typ == 'RECORD':
        fields = field.fields
        assert fields
        names = [el.name for el in fields]
        ibis_types = [_discover_type(el) for el in fields]
        ibis_type = dt.Struct(names, ibis_types)
    else:
        ibis_type = _LEGACY_TO_STANDARD.get(typ, typ)
        ibis_type = _DTYPE_TO_IBIS_TYPE.get(ibis_type, ibis_type)
    if field.mode == 'REPEATED':
        ibis_type = dt.Array(ibis_type)
    return ibis_type


def infer_schema_from_df(df):

    np_dtype_to_field_type = {
        np.dtype('bool'): 'boolean',
        np.dtype('float64'): 'float',
        np.dtype('int64'): 'integer',
        np.dtype('object'): 'string',
    }

    def f(df_dtype_row):
        (name, dtype) = df_dtype_row
        return google.cloud.bigquery.schema.SchemaField(
            name,
            np_dtype_to_field_type[dtype],
            'NULLABLE',
        )

    lst = [f(el) for el in df.dtypes.iteritems()]
    return lst


def bigquery_dtypes_to_ibis_schema(table):
    schema = table.schema
    names = [el.name for el in schema]
    ibis_types = [_discover_type(el) for el in schema]
    pairs = zip(names, ibis_types)
    return ibis.schema(pairs)
