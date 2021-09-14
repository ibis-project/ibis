import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from clickhouse_driver.client import Client as _DriverClient

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base.sql import SQLClient
from ibis.config import options

from .compiler import ClickhouseCompiler

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")
base_typename_re = re.compile(r"(\w+)")


_clickhouse_dtypes = {
    'Null': dt.Null,
    'Nothing': dt.Null,
    'UInt8': dt.UInt8,
    'UInt16': dt.UInt16,
    'UInt32': dt.UInt32,
    'UInt64': dt.UInt64,
    'Int8': dt.Int8,
    'Int16': dt.Int16,
    'Int32': dt.Int32,
    'Int64': dt.Int64,
    'Float32': dt.Float32,
    'Float64': dt.Float64,
    'String': dt.String,
    'FixedString': dt.String,
    'Date': dt.Date,
    'DateTime': dt.Timestamp,
}
_ibis_dtypes = {v: k for k, v in _clickhouse_dtypes.items()}
_ibis_dtypes[dt.String] = 'String'


class ClickhouseDataType:

    __slots__ = 'typename', 'nullable'

    def __init__(self, typename, nullable=False):
        m = base_typename_re.match(typename)
        base_typename = m.groups()[0]
        if base_typename not in _clickhouse_dtypes:
            raise com.UnsupportedBackendType(typename)
        self.typename = base_typename
        self.nullable = nullable

    def __str__(self):
        if self.nullable:
            return f'Nullable({self.typename})'
        else:
            return self.typename

    def __repr__(self):
        return f'<Clickhouse {str(self)}>'

    @classmethod
    def parse(cls, spec):
        # TODO(kszucs): spare parsing, depends on clickhouse-driver#22
        if spec.startswith('Nullable'):
            return cls(spec[9:-1], nullable=True)
        else:
            return cls(spec)

    def to_ibis(self):
        return _clickhouse_dtypes[self.typename](nullable=self.nullable)

    @classmethod
    def from_ibis(cls, dtype, nullable=None):
        typename = _ibis_dtypes[type(dtype)]
        if nullable is None:
            nullable = dtype.nullable
        return cls(typename, nullable=nullable)


@dt.dtype.register(ClickhouseDataType)
def clickhouse_to_ibis_dtype(clickhouse_dtype):
    return clickhouse_dtype.to_ibis()


class ClickhouseTable(ir.TableExpr):
    """References a physical table in Clickhouse"""

    @property
    def _qualified_name(self):
        return self.op().args[0]

    @property
    def _unqualified_name(self):
        return self._match_name()[1]

    @property
    def _client(self):
        return self.op().args[2]

    def _match_name(self):
        m = fully_qualified_re.match(self._qualified_name)
        if not m:
            raise com.IbisError(
                'Cannot determine database name from {}'.format(
                    self._qualified_name
                )
            )
        db, quoted, unquoted = m.groups()
        return db, quoted or unquoted

    @property
    def _database(self):
        return self._match_name()[0]

    def invalidate_metadata(self):
        self._client.invalidate_metadata(self._qualified_name)

    def metadata(self):
        """
        Return parsed results of DESCRIBE FORMATTED statement

        Returns
        -------
        meta : TableMetadata
        """
        return self._client.describe_formatted(self._qualified_name)

    describe_formatted = metadata

    @property
    def name(self):
        return self.op().name

    def insert(self, obj, **kwargs):
        from .identifiers import quote_identifier

        schema = self.schema()

        assert isinstance(obj, pd.DataFrame)
        assert set(schema.names) >= set(obj.columns)

        columns = ', '.join(map(quote_identifier, obj.columns))
        query = 'INSERT INTO {table} ({columns}) VALUES'.format(
            table=self._qualified_name, columns=columns
        )

        # convert data columns with datetime64 pandas dtype to native date
        # because clickhouse-driver 0.0.10 does arithmetic operations on it
        obj = obj.copy()
        for col in obj.select_dtypes(include=[np.datetime64]):
            if isinstance(schema[col], dt.Date):
                obj[col] = obj[col].dt.date

        data = obj.to_dict('records')
        return self._client.con.execute(query, data, **kwargs)


class ClickhouseClient(SQLClient):
    """An Ibis client interface that uses Clickhouse"""

    compiler = ClickhouseCompiler

    def __init__(self, backend, *args, **kwargs):
        self.backend = backend
        self.database_class = backend.database_class
        self.table_class = backend.table_class
        self.table_expr_class = backend.table_expr_class
        self.con = _DriverClient(*args, **kwargs)

    def raw_sql(self, query: str, external_tables={}):
        external_tables_list = []
        for name, df in external_tables.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    'External table is not an instance of pandas ' 'dataframe'
                )
            schema = sch.infer(df)
            external_tables_list.append(
                {
                    'name': name,
                    'data': df.to_dict('records'),
                    'structure': list(
                        zip(
                            schema.names,
                            [
                                str(ClickhouseDataType.from_ibis(t))
                                for t in schema.types
                            ],
                        )
                    ),
                }
            )

        ibis.util.log(query)
        return self.con.execute(
            query,
            columnar=True,
            with_column_types=True,
            external_tables=external_tables_list,
        )

    def ast_schema(self, query_ast, external_tables={}):
        # Allowing signature to accept `external_tables`
        return super().ast_schema(query_ast)

    def fetch_from_cursor(self, cursor, schema):
        data, columns = cursor
        if not len(data):
            # handle empty resultset
            return pd.DataFrame([], columns=schema.names)

        df = pd.DataFrame.from_dict(OrderedDict(zip(schema.names, data)))
        return schema.apply_to(df)

    def close(self):
        """Close Clickhouse connection and drop any temporary objects"""
        self.con.disconnect()

    def _fully_qualified_name(self, name, database):
        if bool(fully_qualified_re.search(name)):
            return name

        database = database or self.current_database
        return f'{database}.`{name}`'

    def get_schema(self, table_name, database=None):
        """
        Return a Schema object for the indicated table and database

        Parameters
        ----------
        table_name : string
          May be fully qualified
        database : string, default None

        Returns
        -------
        schema : ibis Schema
        """
        qualified_name = self._fully_qualified_name(table_name, database)
        query = f'DESC {qualified_name}'
        data, columns = self.raw_sql(query)
        return sch.schema(
            data[0], list(map(ClickhouseDataType.parse, data[1]))
        )

    def set_options(self, options):
        self.con.set_options(options)

    def reset_options(self):
        # Must nuke all cursors
        raise NotImplementedError

    def exists_table(self, name, database=None):
        """
        Determine if the indicated table or view exists

        Parameters
        ----------
        name : string
        database : string, default None

        Returns
        -------
        if_exists : boolean
        """
        return len(self.list_tables(like=name, database=database)) > 0

    def _ensure_temp_db_exists(self):
        name = (options.clickhouse.temp_db,)
        if name not in self.list_databases():
            self.create_database(name, force=True)

    def _get_schema_using_query(self, query, **kwargs):
        data, columns = self.raw_sql(query, **kwargs)
        colnames, typenames = zip(*columns)
        coltypes = list(map(ClickhouseDataType.parse, typenames))
        return sch.schema(colnames, coltypes)

    def _table_command(self, cmd, name, database=None):
        qualified_name = self._fully_qualified_name(name, database)
        return f'{cmd} {qualified_name}'
