import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.file.client import FileClient
from ibis.pandas.core import pre_execute, execute
import pyarrow as pa
import pyarrow.parquet as pq


def connect(dictionary):
        return ParquetClient(dictionary)


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
    'timestamp[ns]': dt.timestamp,
}

_PARQUET_DTYPE_TO_IBIS_TYPE = {
    'UTF8': dt.string,
    'BYTE_ARRAY': dt.binary,
    'TIMESTAMP_MICROS': dt.timestamp,
    'FLOAT': dt.float32,
    'DOUBLE': dt.float64,
    'BOOLEAN': dt.boolean,
    'INT32': dt.int32,
    'INT64': dt.int64,
    'UINT_8': dt.uint8,
    'UINT_16': dt.uint16,
    'UINT_32': dt.uint32,
    'UINT_64': dt.uint64,
}


def arrow_types_to_ibis_schema(schema):
    pairs = []
    for cs in schema:
        column_name = cs.name
        ibis_type = _ARROW_DTYPE_TO_IBIS_TYPE[str(cs.type)]
        pairs.append((column_name, ibis_type))
    return ibis.schema(pairs)


def parquet_types_to_ibis_schema(schema):
    pairs = []
    for cs in schema:
        column_name = cs.name
        try:
            ibis_type = _PARQUET_DTYPE_TO_IBIS_TYPE[cs.logical_type]
        except KeyError:
            ibis_type = _PARQUET_DTYPE_TO_IBIS_TYPE[cs.physical_type]
        pairs.append((column_name, ibis_type))
    return ibis.schema(pairs)


class ParquetTable(ops.DatabaseTable):
    pass


class ParquetClient(FileClient):
    extension = 'parquet'

    def insert(self, path, t, **kwargs):

        path = self.root / path
        df = execute(t)
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
        # tables are files in a dir
        if path is None:
            path = self.root

        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_file():
                    if str(d).endswith(self.extension):
                        tables.append(d.stem)
        elif path.is_file():
            if str(path).endswith(self.extension):
                tables.append(path.stem)
        return tables

    def list_databases(self, path=None):
        # databases are dir
        if path is None:
            path = self.root

        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_dir():
                    tables.append(d.name)
        return tables


@pre_execute.register(ParquetTable, ParquetClient)
def parquet_data_preload_uri_client(op, client, scope=None, **kwargs):

    path = client.dictionary[op.name]
    table = pq.read_table(str(path))
    df = table.to_pandas()
    return {op: df}
