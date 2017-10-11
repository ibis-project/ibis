import re

import ibis
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
from ibis.bigquery import compiler as comp
import toolz
import google.datalab as gd
import google.datalab.bigquery as bq
from ibis.client import (Database, SQLClient)


_fully_qualified_re = re.compile('(.*)\.(.*)|(.*)')


# pip install git+https://github.com/googledatalab/pydatalab
def _parse_bq_maybe_qualified_re(name):
    (dataset, table0, table1) = _fully_qualified_re.match(name).groups()
    table = table0 or table1
    return (dataset, table)


def _ensure_split(table_id, dataset_id):
    (_dataset_id, table_id) = _parse_bq_maybe_qualified_re(table_id)
    assert (dataset_id == _dataset_id) or not (dataset_id and _dataset_id)
    dataset_id = dataset_id or _dataset_id
    return (table_id, dataset_id)


def _bq_table_obj_to_string(table):
    (_, dataset_id, table_id, _) = table.name
    return dataset_id + '.' + table_id


def _bq_make_context(project_id=None):
    context = gd.Context.default()
    if context.project_id != project_id:
        context.set_project_id(project_id)
    return context


@toolz.memoize
def _bq_get_context(project_id):
    return _bq_make_context(project_id)


def _bq_get_datasets(project_id):
    return bq.Datasets(_bq_get_context(project_id))


def _bq_get_dataset(dataset_id, project_id):
    return bq.Dataset(dataset_id, _bq_get_context(project_id))


def _bq_get_table(table_id, dataset_id, project_id):
    (table_id, dataset_id) = _ensure_split(table_id, dataset_id)
    return bq.Table(dataset_id + '.' + table_id, _bq_get_context(project_id))


def _bq_get_schema(table_id, dataset_id, project_id):
    return _bq_get_table(table_id, dataset_id, project_id).schema


class BigQueryDataset(Database):
    pass


class BigQueryClient(SQLClient):

    database_class = BigQueryDataset

    def __init__(self, project_id, dataset_id):
        self.__project_id = project_id
        self.__dataset_id = dataset_id

    @property
    def _project_id(self):
        return self.__project_id

    @property
    def _dataset_id(self):
        return self.__dataset_id

    @property
    def _context(self):
        return _bq_get_context(self._project_id)

    @property
    def _dataset(self):
        return _bq_get_dataset(self._dataset_id, self._project_id)

    def _build_ast(self, expr, params=None):
        return comp.build_ast(expr, params=params)

    def execute(self, expr, limit='default', async=False, params=None,
                output_options=None):
        if limit != 'default' or async or params:
            raise NotImplementedError()

        stmt = expr.compile()
        return (bq.Query(stmt)
                .execute(output_options=output_options, context=self._context)
                .result()
                .to_dataframe()
                )

    @property
    def _table_expr_klass(self):
        return ir.TableExpr

    def _fully_qualified_name(self, table_id, dataset_id=None):
        dataset_id = dataset_id or self._dataset_id
        return dataset_id + '.' + table_id

    def _get_table_schema(self, qualified_name):
        return self.get_schema(qualified_name)

    def list_tables(self, like=None, dataset=None):
        dataset = _bq_get_dataset(dataset or self._dataset_id,
                                  self._project_id)
        result = [table.name.table_id for table in dataset.tables()]
        if like:
            result = [table_name
                      for table_name in result if re.match(like, table_name)]
        return result

    def set_dataset(self, name):
        self.__dataset_id = name

    def exists_database(self, name):
        return _bq_get_dataset(name, self._project_id).exists()

    def list_datasets(self, like=None):
        results = [dataset.name.dataset_id
                   for dataset in _bq_get_datasets(self._project_id)]
        if like:
            results = [dataset_name
                       for dataset_name in results
                       if re.match(like, dataset_name)
                       ]
        return results

    def get_schema(self, name, dataset=None):
        (table_id, dataset_id) = _ensure_split(name, dataset)
        if dataset_id and dataset:
            raise ValueError(
                'Can\'t pass a fully qualified table name *AND* a dataset'
            )
        else:
            dataset_id = dataset or self._dataset_id
        table = _bq_get_table(table_id, dataset_id, self._project_id)
        return bigquery_dtypes_to_ibis_schema(table, None)

    def exists_table(self, name, dataset=None):
        dataset_id = dataset or self._dataset_id
        return _bq_get_table(name, dataset_id, self._project_id).exists()


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


def _discover_type(dct):
    (typ, mode, fields) = (dct.get(k) for k in ('type', 'mode', 'fields'))
    if typ == 'RECORD':
        assert fields
        names = [el['name'] for el in fields]
        ibis_types = [_discover_type(el) for el in fields]
        ibis_type = dt.Struct(names, ibis_types)
    else:
        ibis_type = _LEGACY_TO_STANDARD.get(typ, typ)
        ibis_type = _DTYPE_TO_IBIS_TYPE.get(ibis_type, ibis_type)
    if mode == 'REPEATED':
        ibis_type = dt.Array(ibis_type)
    return ibis_type


def bigquery_dtypes_to_ibis_schema(table, schema=None):
    if schema:
        raise NotImplementedError()
    else:
        schema = dict()

    bq_schema = table.schema._bq_schema
    names = [el['name'] for el in bq_schema]
    ibis_types = [_discover_type(el) for el in bq_schema]
    pairs = zip(names, ibis_types)
    return ibis.schema(pairs)
