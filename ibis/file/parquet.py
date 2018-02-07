import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import ibis.expr.schema as sch
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis.compat import parse_version
from ibis.file.client import FileClient
from ibis.pandas.api import PandasDialect
from ibis.pandas.core import pre_execute, execute


dialect = PandasDialect


# TODO(jreback) complex types are not implemented
_arrow_dtypes = {
    'int8': dt.Int8,
    'int16': dt.Int16,
    'int32': dt.Int32,
    'int64': dt.Int64,
    'uint8': dt.UInt8,
    'uint16': dt.UInt16,
    'uint32': dt.UInt32,
    'uint64': dt.UInt64,
    'halffloat': dt.Float16,
    'float': dt.Float32,
    'double': dt.Float64,
    'string': dt.String,
    'binary': dt.Binary,
    'bool': dt.Boolean,
    'timestamp[ns]': dt.Timestamp,
    'timestamp[us]': dt.Timestamp
}


@dt.dtype.register(pa.DataType)
def pa_dtype(arrow_type, nullable=True):
    return _arrow_dtypes[str(arrow_type)](nullable=nullable)


@sch.infer.register(pq.ParquetSchema)
def infer_parquet_schema(schema):
    pairs = []

    for field in schema.to_arrow_schema():
        ibis_dtype = dt.dtype(field.type, nullable=field.nullable)
        pairs.append((field.name, ibis_dtype))

    return sch.schema(pairs)


def connect(dictionary):
    return ParquetClient(dictionary)


class ParquetTable(ops.DatabaseTable):
    pass


class ParquetClient(FileClient):

    dialect = dialect
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
        schema = sch.infer(parquet_file.schema)

        table = ParquetTable(name, schema, self).to_expr()
        self.dictionary[name] = f

        return table

    def list_tables(self, path=None):
        return self._list_tables_files(path)

    def list_databases(self, path=None):
        return self._list_databases_dirs(path)

    def compile(self, expr, *args, **kwargs):
        return expr

    @property
    def version(self):
        return parse_version(pa.__version__)


@pre_execute.register(ParquetTable, ParquetClient)
def parquet_pre_execute_client(op, client, scope, **kwargs):
    # cache
    if isinstance(scope.get(op), pd.DataFrame):
        return {}

    path = client.dictionary[op.name]
    table = pq.read_table(str(path))
    df = table.to_pandas()
    return {op: df}
