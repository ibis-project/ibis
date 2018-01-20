import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis.file.client import FileClient

from ibis.pandas.api import PandasDialect
from ibis.pandas.core import pre_execute, execute


def connect(dictionary):
    return ParquetClient(dictionary)


# TODO(jreback) complex types are not implemented
_ARROW_DTYPE_TO_IBIS_TYPE = {
    'int8': dt.int8,
    'int16': dt.int16,
    'int32': dt.int32,
    'int64': dt.int64,
    'uint8': dt.uint8,
    'uint16': dt.uint16,
    'uint32': dt.uint32,
    'uint64': dt.uint64,
    'halffloat': dt.float16,
    'float': dt.float32,
    'double': dt.float64,
    'string': dt.string,
    'binary': dt.binary,
    'bool': dt.boolean,
    'timestamp[ns]': dt.timestamp,
    'timestamp[us]': dt.timestamp,
}


def arrow_types_to_ibis_schema(schema):
    pairs = []
    for cs in schema:
        column_name = cs.name
        ibis_type = _ARROW_DTYPE_TO_IBIS_TYPE[str(cs.type)]
        pairs.append((column_name, ibis_type))
    return ibis.schema(pairs)


def parquet_types_to_ibis_schema(schema):
    schema = schema.to_arrow_schema()
    return arrow_types_to_ibis_schema(schema)


class ParquetTable(ops.DatabaseTable):
    pass


class ParquetClient(FileClient):
    dialect = PandasDialect
    extension = 'parquet'

    def insert(self, path, expr, **kwargs):

        path = self.root / path
        df = execute(expr)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path))

    def table(self, name, path):
        if name not in self.list_tables(path):
            raise AttributeError(name)

        if path is None:
            path = self.root

        # get the schema
        f = path / "{}.parquet".format(name)
        parquet_file = pq.ParquetFile(str(f))
        schema = parquet_types_to_ibis_schema(parquet_file.schema)

        t = ParquetTable(name, schema, self).to_expr()
        self.dictionary[name] = f
        return t

    def list_tables(self, path=None):
        return self._list_tables_files(path)

    def list_databases(self, path=None):
        return self._list_databases_dirs(path)

    def compile(self, expr, *args, **kwargs):
        return expr


@pre_execute.register(ParquetTable, ParquetClient)
def parquet_pre_execute_client(op, client, scope, **kwargs):
    # cache
    if isinstance(scope.get(op), pd.DataFrame):
        return {}

    path = client.dictionary[op.name]
    table = pq.read_table(str(path))
    df = table.to_pandas()
    return {op: df}
