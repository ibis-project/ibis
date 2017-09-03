import re
import pandas as pd

import ibis.common as com

from ibis.config import options
from ibis.client import Query, Database, DatabaseEntity, SQLClient

from ibis.clickhouse.compiler import build_ast
from ibis.util import log
from ibis.sql.compiler import DDL
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util

from clickhouse_driver.client import Client as _DriverClient
from toolz import pluck

from .types import PD2CH, CH2PD, CH2IB

# TODO: move to types


fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


class ClickhouseDatabase(Database):
    pass


class ClickhouseQuery(Query):

    def execute(self):
        # synchronous by default
        data, types = self.client._execute(self.compiled_ddl,
                                           with_column_types=True)
        dtypes = [(col, CH2PD[typ]) for col, typ in types]

        # Wes: naive approach, I could use some help to make it more efficient
        df = pd.DataFrame(data, columns=list(pluck(0, dtypes)))
        for col, dtype in dtypes:
            df[col] = df[col].astype(dtype)

        return self._wrap_result(df)

    def _fetch(self, cursor):
        raise NotImplementedError


class ClickhouseClient(SQLClient):
    """An Ibis client interface that uses Clickhouse"""

    database_class = ClickhouseDatabase
    sync_query = ClickhouseQuery

    def __init__(self, *args, **kwargs):
        self.con = _DriverClient(*args, **kwargs)

    def _build_ast(self, expr):
        return build_ast(expr)

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

    def _execute(self, query, with_column_types=False):
        if isinstance(query, DDL):
            query = query.compile()
        self.log(query)

        return self.con.execute(query, with_column_types=with_column_types)

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
            statement += " FROM `{}`".format(database)
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
        data, types = self._execute(query, with_column_types=True)

        names = pluck(0, data)
        ibis_types = map(CH2IB.get, pluck(1, data))

        return dt.Schema(names, ibis_types)

    @property
    def client_options(self):
        return self.con.options

    # def get_options(self):
    #     """
    #     Return current query options for the Clickhouse session
    #     """
    #     query = 'SET'
    #     tuples = self.con.fetchall(query)
    #     return dict(tuples)

    def set_options(self, options):
        self.con.set_options(options)

    def reset_options(self):
        # Must nuke all cursors
        raise NotImplementedError

    def set_compression_codec(self, codec):
        """
        Parameters
        """
        if codec is None:
            codec = 'none'
        else:
            codec = codec.lower()

        if codec not in ('none', 'gzip', 'snappy'):
            raise ValueError('Unknown codec: {0}'.format(codec))

        self.set_options({'COMPRESSION_CODEC': codec})

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

    def _get_concrete_table_path(self, name, database, persist=False):
        if not persist:
            if name is None:
                name = '__ibis_tmp_{0}'.format(util.guid())

            if database is None:
                self._ensure_temp_db_exists()
                database = options.clickhouse.temp_db
            return name, database
        else:
            if name is None:
                raise com.IbisError('Must pass table name if persist=True')
            return name, database

    def _ensure_temp_db_exists(self):
        # TODO: session memoize to avoid unnecessary `SHOW DATABASES` calls
        name = options.clickhouse.temp_db,
        if not self.exists_database(name):
            self.create_database(name, force=True)

    def _wrap_new_table(self, name, database, persist):
        qualified_name = self._fully_qualified_name(name, database)

        if persist:
            t = self.table(qualified_name)
        else:
            schema = self._get_table_schema(qualified_name)
            node = ClickhouseTemporaryTable(qualified_name, schema, self)
            t = self._table_expr_klass(node)

        # Compute number of rows in table for better default query planning
        cardinality = t.count().execute()
        set_card = ("alter table {0} set tblproperties('numRows'='{1}', "
                    "'STATS_GENERATED_VIA_STATS_TASK' = 'true')"
                    .format(qualified_name, cardinality))
        self._execute(set_card)

        self._temp_objects[id(t)] = t

        return t

    def _get_table_schema(self, tname):
        return self.get_schema(tname)

    def _get_schema_using_query(self, query):
        data, types = self._execute(query, with_column_types=True)
        names, clickhouse_types = zip(*types)
        ibis_types = map(CH2IB.get, clickhouse_types)
        return dt.Schema(names, ibis_types)

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


class ClickhouseTemporaryTable(ops.DatabaseTable):

    def __del__(self):
        try:
            self.drop()
        except com.IbisError:
            pass

    def drop(self):
        try:
            self.source.drop_table(self.name)
        except:  # ClickhouseError
            # database might have been dropped
            pass
