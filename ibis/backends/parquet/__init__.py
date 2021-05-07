from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import regex as re
from pkg_resources import parse_version

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base_file import FileClient
from ibis.backends.pandas import PandasDialect
from ibis.backends.pandas.core import execute, execute_node

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
}


@dt.dtype.register(pa.DataType)
def pa_dtype(arrow_type, nullable=True):
    return _arrow_dtypes[str(arrow_type)](nullable=nullable)


@dt.dtype.register(pa.lib.TimestampType)
def pa_timestamp_type(arrow_type, nullable=True):
    return dt.Timestamp(arrow_type.tz, nullable=nullable)


@sch.infer.register(pq.ParquetSchema)
def infer_parquet_schema(schema):
    pairs = []

    for field in schema.to_arrow_schema():
        ibis_dtype = dt.dtype(field.type, nullable=field.nullable)
        name = field.name
        if not re.match(r'^__index_level_\d+__$', name):
            pairs.append((name, ibis_dtype))

    return sch.schema(pairs)


def connect(dictionary):
    return ParquetClient(dictionary)


class ParquetTable(ops.DatabaseTable):
    pass


class ParquetClient(FileClient):

    dialect = dialect
    extension = 'parquet'
    table_class = ParquetTable

    def insert(self, path, expr, **kwargs):
        path = self.root / path
        df = execute(expr)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path))

    def table(self, name: str, path: Optional[str] = None) -> ir.TableExpr:
        if name not in self.list_tables(path):
            raise AttributeError(name)

        if path is None:
            path = self.root

        # get the schema
        f = path / "{}.parquet".format(name)

        parquet_file = pq.ParquetFile(str(f))
        schema = sch.infer(parquet_file.schema)

        table = self.table_class(name, schema, self).to_expr()
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


@execute_node.register(ParquetClient.table_class, ParquetClient)
def parquet_read_table(op, client, scope, **kwargs):
    path = client.dictionary[op.name]
    table = pq.read_table(str(path))
    df = table.to_pandas()
    return df
