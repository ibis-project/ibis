# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import re
# import six
# import threading
# import time
# import weakref
# import traceback

# from posixpath import join as pjoin
# from collections import deque

# import numpy as np
import pandas as pd

import ibis.common as com

from ibis.config import options
from ibis.client import (Query, AsyncQuery, Database,
                         DatabaseEntity, SQLClient)
# from ibis.compat import lzip
from ibis.clickhouse import ddl
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


class ClickhouseDatabase(Database):

    def create_table(self, table_name, obj=None, **kwargs):
        """
        Dispatch to ClickhouseClient.create_table. See that function's
        docstring for more
        """
        return self.client.create_table(table_name, obj=obj,
                                        database=self.name, **kwargs)


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


class ClickhouseAsyncQuery(ClickhouseQuery, AsyncQuery):
    # TODO
    pass


class ClickhouseClient(SQLClient):
    """An Ibis client interface that uses Clickhouse"""

    database_class = ClickhouseDatabase
    sync_query = ClickhouseQuery
    async_query = ClickhouseAsyncQuery

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
        if ddl._is_fully_qualified(name):
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
            m = ddl.fully_qualified_re.match(like)
            if m:
                database, quoted, unquoted = m.groups()
                like = quoted or unquoted
                return self.list_tables(like=like, database=database)
            statement += " LIKE '{0}'".format(like)

        return self._execute(statement)

    # def _get_list(self, cur):
    #     tuples = cur.fetchall()
    #     if len(tuples) > 0:
    #         return list(lzip(*tuples)[0])
    #     else:
    #         return []

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

    def create_database(self, name, path=None, force=False):
        """
        Create a new Clickhouse database

        Parameters
        ----------
        name : string
          Database name
        """
        statement = ddl.CreateDatabase(name, can_exist=force)
        return self._execute(statement)

    def drop_database(self, name, force=False):
        """
        Drop an Clickhouse database

        Parameters
        ----------
        name : string
          Database name
        force : boolean, default False
          If False and there are any tables in this database, raises an
          IntegrityError
        """
        if not force or self.exists_database(name):
            tables = self.list_tables(database=name)
        else:
            tables = []

        if force:
            for table in tables:
                self.log('Dropping {0}'.format('{0}.{1}'.format(name, table)))
                self.drop_table_or_view(table, database=name)
        else:
            if len(tables) > 0:
                raise com.IntegrityError('Database {0} must be empty before '
                                         'being dropped, or set '
                                         'force=True'.format(name))
        statement = ddl.DropDatabase(name, must_exist=not force)
        return self._execute(statement)

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

    def create_view(self, name, expr, database=None):
        """
        Create an Clickhouse view from a table expression

        Parameters
        ----------
        name : string
        expr : ibis TableExpr
        database : string, default None
        """
        ast = self._build_ast(expr)
        select = ast.queries[0]
        statement = ddl.CreateView(name, select, database=database)
        return self._execute(statement)

    def drop_view(self, name, database=None, force=False):
        """
        Drop an Clickhouse view

        Parameters
        ----------
        name : string
        database : string, default None
        force : boolean, default False
          Database may throw exception if table does not exist
        """
        statement = ddl.DropView(name, database=database,
                                 must_exist=not force)
        return self._execute(statement)

    # def create_table(self, table_name, obj=None, schema=None, database=None,
    #                  external=False, force=False,
    #                  # HDFS options
    #                  format='parquet', location=None,
    #                  partition=None, like_parquet=None):
    #     """
    #     Create a new table in CH using an Ibis table expression. This is
    #     currently designed for tables whose data is stored in HDFS (or
    #     eventually other filesystems).

    #     Parameters
    #     ----------
    #     table_name : string
    #     obj : TableExpr or pandas.DataFrame, optional
    #       If passed, creates table from select statement results
    #     schema : ibis.Schema, optional
    #       Mutually exclusive with expr, creates an empty table with a
    #       particular schema
    #     database : string, default None (optional)
    #     force : boolean, default False
    #       Do not create table if table with indicated name already exists
    #     external : boolean, default False
    #       Create an external table; CH will not delete the underlying data
    #       when the table is dropped
    #     format : {'parquet'}
    #     location : string, default None
    #       Specify the directory location where CH reads and writes files
    #       for the table
    #     partition : list of strings
    #       Must pass a schema to use this. Cannot partition from an expression
    #       (create-table-as-select)
    #     like_parquet : string (HDFS path), optional
    #       Can specify in lieu of a schema

    #     Examples
    #     --------
    #     >>> con.create_table('new_table_name', table_expr)  # doctest: +SKIP
    #     """
    #     if like_parquet is not None:
    #         raise NotImplementedError

    #     if obj is not None:
    #         if isinstance(obj, pd.DataFrame):
    #             from ibis.clickh.pandas_interop import write_temp_dataframe
    #             writer, to_insert = write_temp_dataframe(self, obj)
    #         else:
    #             to_insert = obj
    #         ast = self._build_ast(to_insert)
    #         select = ast.queries[0]

    #         statement = ddl.CTAS(table_name, select,
    #                              database=database,
    #                              can_exist=force,
    #                              format=format,
    #                              external=external,
    #                              partition=partition,
    #                              path=location)
    #     elif schema is not None:
    #         statement = ddl.CreateTableWithSchema(
    #             table_name, schema,
    #             database=database,
    #             format=format,
    #             can_exist=force,
    #             external=external,
    #             path=location, partition=partition)
    #     else:
    #         raise com.IbisError('Must pass expr or schema')

    #     return self._execute(statement)

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

    # def insert(self, table_name, obj=None, database=None, overwrite=False,
    #            partition=None, values=None, validate=True):
    #     """
    #     Insert into existing table.

    #     See ClickhouseTable.insert for other parameters.

    #     Parameters
    #     ----------
    #     table_name : string
    #     database : string, default None

    #     Examples
    #     --------
    #     >>> table = 'my_table'
    #     >>> con.insert(table, table_expr)  # doctest: +SKIP

    #     # Completely overwrite contents
    #     >>> con.insert(table, table_expr, overwrite=True)  # doctest: +SKIP
    #     """
    #     table = self.table(table_name, database=database)
    #     return table.insert(obj=obj, overwrite=overwrite,
    #                         partition=partition,
    #                         values=values, validate=validate)

    # def load_data(self, table_name, path, database=None, overwrite=False,
    #               partition=None):
    #     """
    #     Wraps the LOAD DATA DDL statement.
    #     Loads data into an Clickhouse table by physically moving data files.

    #     Parameters
    #     ----------
    #     table_name : string
    #     database : string, default None (optional)
    #     """
    #     table = self.table(table_name, database=database)
    #     return table.load_data(path, overwrite=overwrite,
    #                            partition=partition)

    # def drop_table(self, table_name, database=None, force=False):
    #     """
    #     Drop an Clickhouse table

    #     Parameters
    #     ----------
    #     table_name : string
    #     database : string, default None (optional)
    #     force : boolean, default False
    #       Database may throw exception if table does not exist

    #     Examples
    #     --------
    #     >>> table = 'my_table'
    #     >>> db = 'operations'
    #     >>> con.drop_table(table, database=db, force=True)  # doctest: +SKIP
    #     """
    #     statement = ddl.DropTable(table_name, database=database,
    #                               must_exist=not force)
    #     self._execute(statement)

    # def drop_table_or_view(self, name, database=None, force=False):
    #     """
    #     Attempt to drop a relation that may be a view or table
    #     """
    #     try:
    #         self.drop_table(name, database=database)
    #     except Exception as e:
    #         try:
    #             self.drop_view(name, database=database)
    #         except Exception:
    #             raise e

    def _get_table_schema(self, tname):
        return self.get_schema(tname)

    def _get_schema_using_query(self, query):
        data, types = self._execute(query, with_column_types=True)
        names, clickhouse_types = zip(*types)
        ibis_types = map(CH2IB.get, clickhouse_types)
        return dt.Schema(names, ibis_types)

    # def describe_formatted(self, name, database=None):
    #     """
    #     Retrieve results of DESCRIBE FORMATTED command. See Clickhouse
    #     documentation for more.

    #     Parameters
    #     ----------
    #     name : string
    #       Table name. Can be fully qualified (with database)
    #     database : string, optional
    #     """
    #     from ibis.clickhouse.metadata import parse_metadata

    #     stmt = self._table_command('DESCRIBE FORMATTED',
    #                                name, database=database)
    #     query = ClickhouseQuery(self, stmt)
    #     result = query.execute()

    #     # Leave formatting to pandas
    #     for c in result.columns:
    #         result[c] = result[c].str.strip()

    #     return parse_metadata(result)

    # TODO: detatch/drop/attach/freeze/fetch partition
    # def list_partitions(self, name, database=None):
    #     stmt = self._table_command('SHOW PARTITIONS', name,
    #                                database=database)
    #     return self._exec_statement(stmt)

    def _exec_statement(self, stmt, adapter=None):
        query = ClickhouseQuery(self, stmt)
        result = query.execute()
        if adapter is not None:
            result = adapter(result)
        return result

    def _table_command(self, cmd, name, database=None):
        qualified_name = self._fully_qualified_name(name, database)
        return '{0} {1}'.format(cmd, qualified_name)

    # def write_dataframe(self, df, path, format='csv', async=False):
    #     """
    #     Write a pandas DataFrame to indicated file path in the
    #     indicated format

    #     Parameters
    #     ----------
    #     df : DataFrame
    #     path : string
    #       Absolute output path
    #     format : {'csv'}, default 'csv'
    #     async : boolean, default False
    #       Not yet supported

    #     Returns
    #     -------
    #     None (for now)
    #     """
    #     from ibis.clickhouse.pandas_interop import DataFrameWriter

    #     if async:
    #         raise NotImplementedError

    #     writer = DataFrameWriter(self, df)
    #     return writer.write_csv(path)


# ----------------------------------------------------------------------
# ORM-ish usability layer


class ScalarFunction(DatabaseEntity):

    def drop(self):
        pass


class AggregateFunction(DatabaseEntity):

    def drop(self):
        pass


# TODO: table engines: MergeTree, SummingMergeTree etc.
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
        m = ddl.fully_qualified_re.match(self._qualified_name)
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

    def drop(self):
        """Drop the table from the database"""
        self._client.drop_table_or_view(self._qualified_name)

    # def insert(self, obj=None, overwrite=False, partition=None,
    #            values=None, validate=True):
    #     """
    #     Insert into Clickhouse table. Wraps ClickhouseClient.insert

    #     Parameters
    #     ----------
    #     obj : TableExpr or pandas DataFrame
    #     overwrite : boolean, default False
    #       If True, will replace existing contents of table
    #     partition : list or dict, optional
    #       For partitioned tables, indicate the partition that's being inserte
    #       into, either with an ordered list of partition keys or a dict of
    #       partition field name to value. For example for the partition
    #       (year=2007, month=7), this can be either (2007, 7) or {'year': 2007
    #       'month': 7}.
    #     validate : boolean, default True
    #       If True, do more rigorous validation that schema of table being
    #       inserted is compatible with the existing table

    #     Examples
    #     --------
    #     >>> t.insert(table_expr)  # doctest: +SKIP

    #     # Completely overwrite contents
    #     >>> t.insert(table_expr, overwrite=True)  # doctest: +SKIP
    #     """
    #     if isinstance(obj, pd.DataFrame):
    #         from ibis.clickhouse.pandas_interop import write_temp_dataframe
    #         writer, expr = write_temp_dataframe(self._client, obj)
    #     else:
    #         expr = obj

    #     if values is not None:
    #         raise NotImplementedError

    #     if validate:
    #         existing_schema = self.schema()
    #         insert_schema = expr.schema()
    #         if not insert_schema.equals(existing_schema):
    #             _validate_compatible(insert_schema, existing_schema)

    #     if partition is not None:
    #         partition_schema = self.partition_schema()
    #         partition_schema_names = frozenset(partition_schema.names)
    #         expr = expr.projection([
    #             column for column in expr.columns
    #             if column not in partition_schema_names
    #         ])
    #     else:
    #         partition_schema = None

    #     ast = build_ast(expr)
    #     select = ast.queries[0]
    #     statement = ddl.InsertSelect(self._qualified_name,
    #                                  select,
    #                                  partition=partition,
    #                                  partition_schema=partition_schema,
    #                                  overwrite=overwrite)
    #     return self._execute(statement)

    # def load_data(self, path, overwrite=False, partition=None):
    #     """
    #     Wraps the LOAD DATA DDL statement. Loads data into an CH table by
    #     physically moving data files.

    #     Parameters
    #     ----------
    #     path : string
    #     overwrite : boolean, default False
    #       Overwrite the existing data in the entire table or indicated
    #       partition
    #     partition : dict, optional
    #       If specified, the partition must already exist

    #     Returns
    #     -------
    #     query : ClickhouseQuery
    #     """
    #     if partition is not None:
    #         partition_schema = self.partition_schema()
    #     else:
    #         partition_schema = None

    #     stmt = ddl.LoadData(self._qualified_name, path,
    #                         partition=partition,
    #                         partition_schema=partition_schema)

    #     return self._execute(stmt)

    @property
    def name(self):
        return self.op().name

    def rename(self, new_name, database=None):
        """Rename table inside Clickhouse.

        References to the old table are no longer valid.

        Parameters
        ----------
        new_name : string
        database : string

        Returns
        -------
        renamed : ClickhouseTable
        """
        m = ddl.fully_qualified_re.match(new_name)
        if not m and database is None:
            database = self._database
        statement = ddl.RenameTable(self._qualified_name, new_name,
                                    new_database=database)
        self._client._execute(statement)

        op = self.op().change_name(statement.new_qualified_name)
        return ClickhouseTable(op)

    def _execute(self, stmt):
        return self._client._execute(stmt)

    # @property
    # def is_partitioned(self):
    #     """
    #     True if the table is partitioned
    #     """
    #     return self.metadata().is_partitioned

    # def partition_schema(self):
    #     """
    #     For partitioned tables, return the schema (names and types) for the
    #     partition columns

    #     Returns
    #     -------
    #     partition_schema : ibis Schema
    #     """
    #     schema = self.schema()
    #     name_to_type = dict(zip(schema.names, schema.types))

    #     result = self.partitions()

    #     partition_fields = []
    #     for x in result.columns:
    #         if x not in name_to_type:
    #             break
    #         partition_fields.append((x, name_to_type[x]))

    #     pnames, ptypes = zip(*partition_fields)
    #     return dt.Schema(pnames, ptypes)

    # def add_partition(self, spec, location=None):
    #     """
    #     Add a new table partition, creating any new directories in HDFS if
    #     necessary.

    #     Partition parameters can be set in a single DDL statement, or you can
    #     use alter_partition to set them after the fact.

    #     Returns
    #     -------
    #     None (for now)
    #     """
    #     part_schema = self.partition_schema()
    #     stmt = ddl.AddPartition(self._qualified_name, spec, part_schema,
    #                             location=location)
    #     return self._execute(stmt)

    def alter(self, location=None, format=None, tbl_properties=None,
              serde_properties=None):
        """
        Change setting and parameters of the table.

        Parameters
        ----------
        location : string, optional
          For partitioned tables, you may want the alter_partition function
        format : string, optional
        tbl_properties : dict, optional
        serde_properties : dict, optional

        Returns
        -------
        None (for now)
        """
        def _run_ddl(**kwds):
            stmt = ddl.AlterTable(self._qualified_name, **kwds)
            return self._execute(stmt)

        return self._alter_table_helper(_run_ddl, location=location,
                                        format=format,
                                        tbl_properties=tbl_properties,
                                        serde_properties=serde_properties)

    # def alter_partition(self, spec, location=None, format=None,
    #                     tbl_properties=None,
    #                     serde_properties=None):
    #     """
    #     Change setting and parameters of an existing partition

    #     Parameters
    #     ----------
    #     spec : dict or list
    #       The partition keys for the partition being modified
    #     location : string, optional
    #     format : string, optional
    #     tbl_properties : dict, optional
    #     serde_properties : dict, optional

    #     Returns
    #     -------
    #     None (for now)
    #     """
    #     part_schema = self.partition_schema()

    #     def _run_ddl(**kwds):
    #         stmt = ddl.AlterPartition(self._qualified_name, spec,
    #                                   part_schema, **kwds)
    #         return self._execute(stmt)

    #     return self._alter_table_helper(_run_ddl, location=location,
    #                                     format=format,
    #                                     tbl_properties=tbl_properties,
    #                                     serde_properties=serde_properties)

    # def _alter_table_helper(self, f, **alterations):
    #     results = []
    #     for k, v in alterations.items():
    #         if v is None:
    #             continue
    #         result = f(**{k: v})
    #         results.append(result)
    #     return results

    # def drop_partition(self, spec):
    #     """
    #     Drop an existing table partition
    #     """
    #     part_schema = self.partition_schema()
    #     stmt = ddl.DropPartition(self._qualified_name, spec, part_schema)
    #     return self._execute(stmt)

    # def partitions(self):
    #     """
    #     Return a pandas.DataFrame giving information about this table's
    #     partitions. Raises an exception if the table is not partitioned.

    #     Returns
    #     -------
    #     partitions : pandas.DataFrame
    #     """
    #     return self._client.list_partitions(self._qualified_name)


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


def _validate_compatible(from_schema, to_schema):
    if set(from_schema.names) != set(to_schema.names):
        raise com.IbisInputError('Schemas have different names')

    for name in from_schema:
        lt = from_schema[name]
        rt = to_schema[name]
        if not rt.can_implicit_cast(lt):
            raise com.IbisInputError('Cannot safely cast {0!r} to {1!r}'
                                     .format(lt, rt))
