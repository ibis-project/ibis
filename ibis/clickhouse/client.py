import re
import pandas as pd

import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.expr.lineage as lin
import ibis.expr.operations as ops

from ibis.config import options
from ibis.compat import zip as czip
from ibis.client import Query, Database, DatabaseEntity, SQLClient
from ibis.clickhouse.compiler import build_ast, ClickhouseDialect
from ibis.util import log
from ibis.sql.compiler import DDL

from clickhouse_driver.client import Client as _DriverClient

from .types import clickhouse_to_pandas, clickhouse_to_ibis, ibis_to_clickhouse


fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


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
            types = ibis_types_to_clickhouse_types(table.schema.types)
            struct = list(zip(table.schema.names, types))
            tables.append(dict(name=table.name, data=data, structure=struct))

        return tables

    def execute(self):
        cursor = self.client._execute(
            self.compiled_ddl,
            external_tables=self._external_tables()
        )
        result = self._fetch(cursor)
        return self._wrap_result(result)

    def _fetch(self, cursor):
        data, columns = cursor
        names, types = czip(*columns)

        cols = {}
        for (col, name, db_type) in czip(data, names, types):
            dtype = self._db_type_to_dtype(db_type, name)
            try:
                cols[name] = pd.Series(col, dtype=dtype)
            except TypeError:
                cols[name] = pd.Series(col)

        return pd.DataFrame(cols, columns=names)

    def _db_type_to_dtype(self, db_type, column):
        try:
            return clickhouse_to_pandas[db_type]
        except KeyError:
            return com.UnsupportedBackendType(db_type)


def clickhouse_types_to_ibis_types(types):
    result = []

    for t in types:
        try:
            value = clickhouse_to_ibis[t]
        except KeyError:
            raise com.UnsupportedBackendType(t)
        else:
            result.append(value)
    return result


def ibis_types_to_clickhouse_types(types):
    result = []

    for t in types:
        try:
            value = ibis_to_clickhouse[str(t)]
        except KeyError:
            raise com.UnsupportedBackendType(t)
        else:
            result.append(value)
    return result


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

    def _execute(self, query, external_tables=[]):
        if isinstance(query, DDL):
            query = query.compile()
        self.log(query)

        return self.con.execute(query, columnar=True, with_column_types=True,
                                external_tables=external_tables)

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
        data, _ = self._execute(query)

        names, types = data[:2]
        ibis_types = clickhouse_types_to_ibis_types(types)
        try:
            ibis_types = map(clickhouse_to_ibis.__getitem__, types)
        except KeyError:
            raise com.UnsupportedBackendType()

        return sch.Schema(names, ibis_types)

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
        _, types = self._execute(query)
        names, clickhouse_types = zip(*types)
        ibis_types = clickhouse_types_to_ibis_types(clickhouse_types)
        return sch.Schema(names, ibis_types)

    def _exec_statement(self, stmt, adapter=None):
        query = ClickhouseQuery(self, stmt)
        result = query.execute()
        if adapter is not None:
            result = adapter(result)
        return result

    def _table_command(self, cmd, name, database=None):
        qualified_name = self._fully_qualified_name(name, database)
        return '{0} {1}'.format(cmd, qualified_name)


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
