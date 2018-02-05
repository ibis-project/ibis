import re
import pandas as pd

import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.expr.lineage as lin
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis.config import options
from ibis.compat import zip as czip, parse_version
from ibis.client import Query, Database, DatabaseEntity, SQLClient
from ibis.clickhouse.compiler import ClickhouseDialect, build_ast
from ibis.util import log
from ibis.sql.compiler import DDL

from clickhouse_driver.client import Client as _DriverClient


fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


_clickhouse_dtypes = {
    'Null': dt.Null,
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
    'DateTime': dt.Timestamp
}
_ibis_dtypes = {v: k for k, v in _clickhouse_dtypes.items()}
_ibis_dtypes[dt.String] = 'String'


class ClickhouseDataType(object):

    __slots__ = 'typename', 'nullable'

    def __init__(self, typename, nullable=False):
        if typename not in _clickhouse_dtypes:
            raise com.UnsupportedBackendType(typename)
        self.typename = typename
        self.nullable = nullable

    def __str__(self):
        if self.nullable:
            return 'Nullable({})'.format(self.typename)
        else:
            return self.typename

    def __repr__(self):
        return '<Clickhouse {}>'.format(str(self))

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


class ClickhouseDatabase(Database):
    pass


class ClickhouseQuery(Query):

    def _external_tables(self):
        table_set = getattr(self.ddl, 'table_set', None)
        query_params = getattr(self.ddl, 'params', {})

        if table_set is None:
            return []

        tables = []
        for table in lin.roots(table_set, ClickhouseExternalTable):
            data = table.execute(params=query_params).to_dict('records')

            typenames = [str(ClickhouseDataType.from_ibis(dtype))
                         for dtype in table.schema.types]
            structure = list(zip(table.schema.names, typenames))

            tables.append(dict(data=data,
                               name=table.name,
                               structure=structure))

        return tables

    def execute(self):
        cursor = self.client._execute(
            self.compiled_ddl,
            external_tables=self._external_tables()
        )
        result = self._fetch(cursor)
        return self._wrap_result(result)

    def _fetch(self, cursor):
        data, colnames, coltypes = cursor
        schema = sch.schema(colnames, coltypes)
        # TODO(kszucs): schema = self.schema() instead

        columns = {}
        for (column, (name, dtype)) in czip(data, schema.to_pandas()):
            try:
                columns[name] = pd.Series(column, dtype=dtype)
            except TypeError:
                columns[name] = pd.Series(column)

        return pd.DataFrame(columns, columns=schema.names)


class ClickhouseClient(SQLClient):
    """An Ibis client interface that uses Clickhouse"""

    database_class = ClickhouseDatabase
    sync_query = ClickhouseQuery
    dialect = ClickhouseDialect

    def __init__(self, *args, **kwargs):
        self.con = _DriverClient(*args, **kwargs)

    def _build_ast(self, expr, params=None):
        return build_ast(expr, params=params)

    @property
    def current_database(self):
        # might be better to use driver.Connection instead of Client
        return self.con.connection.database

    @property
    def _table_expr_klass(self):
        return ClickhouseTable

    def log(self, msg):
        log(msg)

    def close(self):
        """Close Clickhouse connection and drop any temporary objects"""
        self.con.disconnect()

    def _execute(self, query, external_tables=(), results=True):
        if isinstance(query, DDL):
            query = query.compile()
        self.log(query)

        response = self.con.process_ordinary_query(
            query, columnar=True, with_column_types=True,
            external_tables=external_tables
        )
        if not results:
            return response

        data, columns = response
        colnames, typenames = czip(*columns)
        coltypes = list(map(ClickhouseDataType.parse, typenames))

        return data, colnames, coltypes

    def _fully_qualified_name(self, name, database):
        if bool(fully_qualified_re.search(name)):
            return name

        database = database or self.current_database
        return '{0}.`{1}`'.format(database, name)

    def list_tables(self, like=None, database=None):
        """
        List tables in the current (or indicated) database. Like the SHOW
        TABLES command in the clickhouse-shell.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'
        database : string, default None
          If not passed, uses the current/default database

        Returns
        -------
        tables : list of strings
        """
        statement = 'SHOW TABLES'
        if database:
            statement += " FROM `{0}`".format(database)
        if like:
            m = fully_qualified_re.match(like)
            if m:
                database, quoted, unquoted = m.groups()
                like = quoted or unquoted
                return self.list_tables(like=like, database=database)
            statement += " LIKE '{0}'".format(like)

        return self._execute(statement)

    def set_database(self, name):
        """
        Set the default database scope for client
        """
        self.con.database = name

    def exists_database(self, name):
        """
        Checks if a given database exists

        Parameters
        ----------
        name : string
          Database name

        Returns
        -------
        if_exists : boolean
        """
        return len(self.list_databases(like=name)) > 0

    def list_databases(self, like=None):
        """
        List databases in the Clickhouse cluster.
        Like the SHOW DATABASES command in the clickhouse-shell.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'

        Returns
        -------
        databases : list of strings
        """
        statement = 'SELECT name FROM system.databases'
        if like:
            statement += " WHERE name LIKE '{0}'".format(like)

        return self._execute(statement)

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
        query = 'DESC {0}'.format(qualified_name)
        data = self._execute(query)[0]

        colnames, coltypes = data[:2]
        coltypes = list(map(ClickhouseDataType.parse, coltypes))

        return sch.schema(colnames, coltypes)

    @property
    def client_options(self):
        return self.con.options

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
        name = options.clickhouse.temp_db,
        if not self.exists_database(name):
            self.create_database(name, force=True)

    def _get_table_schema(self, tname):
        return self.get_schema(tname)

    def _get_schema_using_query(self, query):
        _, colnames, coltypes = self._execute(query)
        return sch.schema(colnames, coltypes)

    def _exec_statement(self, stmt, adapter=None):
        query = ClickhouseQuery(self, stmt)
        result = query.execute()
        if adapter is not None:
            result = adapter(result)
        return result

    def _table_command(self, cmd, name, database=None):
        qualified_name = self._fully_qualified_name(name, database)
        return '{0} {1}'.format(cmd, qualified_name)

    @property
    def version(self):
        self.con.connection.force_connect()

        try:
            server = self.con.connection.server_info
            vstring = '{}.{}.{}'.format(server.version_major,
                                        server.version_minor,
                                        server.revision)
        except Exception:
            self.con.connection.disconnect()
            raise
        else:
            return parse_version(vstring)


class ClickhouseTable(ir.TableExpr, DatabaseEntity):
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
            raise com.IbisError('Cannot determine database name from {0}'
                                .format(self._qualified_name))
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

    def _execute(self, stmt):
        return self._client._execute(stmt)

    def insert(self, obj, **kwargs):
        from .identifiers import quote_identifier
        schema = self.schema()

        assert isinstance(obj, pd.DataFrame)
        assert set(schema.names) >= set(obj.columns)

        columns = ', '.join(map(quote_identifier, obj.columns))
        query = 'INSERT INTO {table} ({columns}) VALUES'.format(
            table=self._qualified_name, columns=columns)

        data = obj.to_dict('records')
        return self._client.con.process_insert_query(query, data, **kwargs)


class ClickhouseExternalTable(ops.DatabaseTable):

    def execute(self, *args, **kwargs):
        return self.source.execute(*args, **kwargs)

    def root_tables(self):
        return [self]


def external_table(name, table):
    """
    Flag a Selection or a DatabaseTable as external to Clickhouse

    Parameters
    ----------
    table : TableExpr

    Returns
    -------
    table : ClickhouseExternalTable
    """
    op = ClickhouseExternalTable(name, table.schema(), table)
    return ClickhouseTable(op)
