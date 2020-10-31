import io
import operator
import re
import threading
import time
import traceback
import weakref
from collections import deque
from posixpath import join as pjoin

import numpy as np
import pandas as pd
from pkg_resources import parse_version

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base_sql.ddl import (
    CTAS,
    AlterTable,
    CreateDatabase,
    CreateTableWithSchema,
    CreateView,
    DropDatabase,
    DropTable,
    DropView,
    InsertSelect,
    RenameTable,
    TruncateTable,
    fully_qualified_re,
    is_fully_qualified,
)
from ibis.backends.base_sqlalchemy.compiler import DDL, DML
from ibis.client import Database, DatabaseEntity, Query, SQLClient
from ibis.config import options
from ibis.filesystems import HDFS, WebHDFS
from ibis.util import log

from . import ddl, udf
from .compat import HS2Error, ImpylaError, impyla
from .compiler import ImpalaDialect, build_ast


class ImpalaDatabase(Database):
    def create_table(self, table_name, obj=None, **kwargs):
        """
        Dispatch to ImpalaClient.create_table. See that function's docstring
        for more
        """
        return self.client.create_table(
            table_name, obj=obj, database=self.name, **kwargs
        )

    def list_udfs(self, like=None):
        return self.client.list_udfs(
            like=self._qualify_like(like), database=self.name
        )

    def list_udas(self, like=None):
        return self.client.list_udas(
            like=self._qualify_like(like), database=self.name
        )


class ImpalaConnection:

    """
    Database connection wrapper
    """

    def __init__(self, pool_size=8, database='default', **params):
        self.params = params
        self.database = database

        self.lock = threading.Lock()

        self.options = {}

        self.max_pool_size = pool_size
        self._connections = weakref.WeakSet()

        self.connection_pool = deque(maxlen=pool_size)
        self.connection_pool_size = 0

    def set_options(self, options):
        self.options.update(options)

    def close(self):
        """
        Close all open Impyla sessions
        """
        for impyla_connection in self._connections:
            impyla_connection.close()

        self._connections.clear()
        self.connection_pool.clear()

    def set_database(self, name):
        self.database = name

    def disable_codegen(self, disabled=True):
        key = 'DISABLE_CODEGEN'
        if disabled:
            self.options[key] = '1'
        elif key in self.options:
            del self.options[key]

    def execute(self, query):
        if isinstance(query, (DDL, DML)):
            query = query.compile()

        cursor = self._get_cursor()
        self.log(query)

        try:
            cursor.execute(query)
        except Exception:
            cursor.release()
            self.error(
                'Exception caused by {}: {}'.format(
                    query, traceback.format_exc()
                )
            )
            raise

        return cursor

    def log(self, msg):
        log(msg)

    def error(self, msg):
        self.log(msg)

    def fetchall(self, query):
        with self.execute(query) as cur:
            results = cur.fetchall()
        return results

    def _get_cursor(self):
        try:
            cursor = self.connection_pool.popleft()
        except IndexError:  # deque is empty
            if self.connection_pool_size < self.max_pool_size:
                return self._new_cursor()
            raise com.InternalError('Too many concurrent / hung queries')
        else:
            if (
                cursor.database != self.database
                or cursor.options != self.options
            ):
                return self._new_cursor()
            cursor.released = False
            return cursor

    def _new_cursor(self):
        params = self.params.copy()
        con = impyla.connect(database=self.database, **params)

        self._connections.add(con)

        # make sure the connection works
        cursor = con.cursor(user=params.get('user'), convert_types=True)
        cursor.ping()

        wrapper = ImpalaCursor(
            cursor, self, con, self.database, self.options.copy()
        )
        wrapper.set_options()
        return wrapper

    def ping(self):
        self._get_cursor()._cursor.ping()

    def release(self, cur):
        self.connection_pool.append(cur)


class ImpalaCursor:
    def __init__(self, cursor, con, impyla_con, database, options):
        self._cursor = cursor
        self.con = con
        self.impyla_con = impyla_con
        self.database = database
        self.options = options
        self.released = False
        self.con.connection_pool_size += 1

    def __del__(self):
        try:
            self._close_cursor()
        except Exception:
            pass

        with self.con.lock:
            self.con.connection_pool_size -= 1

    def _close_cursor(self):
        try:
            self._cursor.close()
        except HS2Error as e:
            # connection was closed elsewhere
            already_closed_messages = [
                'invalid query handle',
                'invalid session',
            ]
            for message in already_closed_messages:
                if message in e.args[0].lower():
                    break
            else:
                raise

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def set_options(self):
        for k, v in self.options.items():
            query = 'SET {} = {!r}'.format(k, v)
            self._cursor.execute(query)

    @property
    def description(self):
        return self._cursor.description

    def release(self):
        if not self.released:
            self.con.release(self)
            self.released = True

    def execute(self, stmt):
        self._cursor.execute_async(stmt)
        self._wait_synchronous()

    def _wait_synchronous(self):
        # Wait to finish, but cancel if KeyboardInterrupt
        from impala.hiveserver2 import OperationalError

        loop_start = time.time()

        def _sleep_interval(start_time):
            elapsed = time.time() - start_time
            if elapsed < 0.05:
                return 0.01
            elif elapsed < 1.0:
                return 0.05
            elif elapsed < 10.0:
                return 0.1
            elif elapsed < 60.0:
                return 0.5
            return 1.0

        cur = self._cursor
        try:
            while True:
                state = cur.status()
                if self._cursor._op_state_is_error(state):
                    raise OperationalError("Operation is in ERROR_STATE")
                if not cur._op_state_is_executing(state):
                    break
                time.sleep(_sleep_interval(loop_start))
        except KeyboardInterrupt:
            print('Canceling query')
            self.cancel()
            raise

    def is_finished(self):
        return not self.is_executing()

    def is_executing(self):
        return self._cursor.is_executing()

    def cancel(self):
        self._cursor.cancel_operation()

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self, columnar=False):
        if columnar:
            return self._cursor.fetchcolumnar()
        else:
            return self._cursor.fetchall()


class ImpalaQuery(Query):
    def _fetch(self, cursor):
        batches = cursor.fetchall(columnar=True)
        names = [x[0] for x in cursor.description]
        df = _column_batches_to_dataframe(names, batches)

        # Ugly Hack for PY2 to ensure unicode values for string columns
        if self.expr is not None:
            # in case of metadata queries there is no expr and
            # self.schema() would raise an exception
            return self.schema().apply_to(df)

        return df


def _column_batches_to_dataframe(names, batches):
    cols = {}
    for name, chunks in zip(names, zip(*[b.columns for b in batches])):
        cols[name] = _chunks_to_pandas_array(chunks)
    return pd.DataFrame(cols, columns=names)


def _chunks_to_pandas_array(chunks):
    total_length = 0
    have_nulls = False
    for c in chunks:
        total_length += len(c)
        have_nulls = have_nulls or c.nulls.any()

    type_ = chunks[0].data_type
    numpy_type = _HS2_TTypeId_to_dtype[type_]

    def fill_nonnull(target, chunks):
        pos = 0
        for c in chunks:
            target[pos : pos + len(c)] = c.values
            pos += len(c.values)

    def fill(target, chunks, na_rep):
        pos = 0
        for c in chunks:
            nulls = c.nulls.copy()
            nulls.bytereverse()
            bits = np.frombuffer(nulls.tobytes(), dtype='u1')
            mask = np.unpackbits(bits).view(np.bool_)

            k = len(c)

            dest = target[pos : pos + k]
            dest[:] = c.values
            dest[mask[:k]] = na_rep

            pos += k

    if have_nulls:
        if numpy_type in ('bool', 'datetime64[ns]'):
            target = np.empty(total_length, dtype='O')
            na_rep = np.nan
        elif numpy_type.startswith('int'):
            target = np.empty(total_length, dtype='f8')
            na_rep = np.nan
        else:
            target = np.empty(total_length, dtype=numpy_type)
            na_rep = np.nan

        fill(target, chunks, na_rep)
    else:
        target = np.empty(total_length, dtype=numpy_type)
        fill_nonnull(target, chunks)

    return target


_HS2_TTypeId_to_dtype = {
    'BOOLEAN': 'bool',
    'TINYINT': 'int8',
    'SMALLINT': 'int16',
    'INT': 'int32',
    'BIGINT': 'int64',
    'TIMESTAMP': 'datetime64[ns]',
    'FLOAT': 'float32',
    'DOUBLE': 'float64',
    'STRING': 'object',
    'DECIMAL': 'object',
    'BINARY': 'object',
    'VARCHAR': 'object',
    'CHAR': 'object',
}


class ImpalaDatabaseTable(ops.DatabaseTable):
    pass


class ImpalaTable(ir.TableExpr, DatabaseEntity):

    """
    References a physical table in the Impala-Hive metastore
    """

    @property
    def _qualified_name(self):
        return self.op().args[0]

    @property
    def _unqualified_name(self):
        return self._match_name()[1]

    @property
    def _client(self):
        return self.op().source

    def _match_name(self):
        m = fully_qualified_re.match(self._qualified_name)
        if not m:
            raise com.IbisError(
                'Cannot determine database name from {0}'.format(
                    self._qualified_name
                )
            )
        db, quoted, unquoted = m.groups()
        return db, quoted or unquoted

    @property
    def _database(self):
        return self._match_name()[0]

    def compute_stats(self, incremental=False):
        """
        Invoke Impala COMPUTE STATS command to compute column, table, and
        partition statistics.

        See also ImpalaClient.compute_stats
        """
        return self._client.compute_stats(
            self._qualified_name, incremental=incremental
        )

    def invalidate_metadata(self):
        self._client.invalidate_metadata(self._qualified_name)

    def refresh(self):
        self._client.refresh(self._qualified_name)

    def metadata(self):
        """
        Return parsed results of DESCRIBE FORMATTED statement

        Returns
        -------
        meta : TableMetadata
        """
        return self._client.describe_formatted(self._qualified_name)

    describe_formatted = metadata

    def files(self):
        """
        Return results of SHOW FILES statement
        """
        return self._client.show_files(self._qualified_name)

    def drop(self):
        """
        Drop the table from the database
        """
        self._client.drop_table_or_view(self._qualified_name)

    def truncate(self):
        self._client.truncate_table(self._qualified_name)

    def insert(
        self,
        obj=None,
        overwrite=False,
        partition=None,
        values=None,
        validate=True,
    ):
        """
        Insert into Impala table. Wraps ImpalaClient.insert

        Parameters
        ----------
        obj : TableExpr or pandas DataFrame
        overwrite : boolean, default False
          If True, will replace existing contents of table
        partition : list or dict, optional
          For partitioned tables, indicate the partition that's being inserted
          into, either with an ordered list of partition keys or a dict of
          partition field name to value. For example for the partition
          (year=2007, month=7), this can be either (2007, 7) or {'year': 2007,
          'month': 7}.
        validate : boolean, default True
          If True, do more rigorous validation that schema of table being
          inserted is compatible with the existing table

        Examples
        --------
        >>> t.insert(table_expr)  # doctest: +SKIP

        # Completely overwrite contents
        >>> t.insert(table_expr, overwrite=True)  # doctest: +SKIP
        """
        if isinstance(obj, pd.DataFrame):
            from .pandas_interop import write_temp_dataframe

            writer, expr = write_temp_dataframe(self._client, obj)
        else:
            expr = obj

        if values is not None:
            raise NotImplementedError

        if validate:
            existing_schema = self.schema()
            insert_schema = expr.schema()
            if not insert_schema.equals(existing_schema):
                _validate_compatible(insert_schema, existing_schema)

        if partition is not None:
            partition_schema = self.partition_schema()
            partition_schema_names = frozenset(partition_schema.names)
            expr = expr.projection(
                [
                    column
                    for column in expr.columns
                    if column not in partition_schema_names
                ]
            )
        else:
            partition_schema = None

        ast = build_ast(expr, ImpalaDialect.make_context())
        select = ast.queries[0]
        statement = InsertSelect(
            self._qualified_name,
            select,
            partition=partition,
            partition_schema=partition_schema,
            overwrite=overwrite,
        )
        return self._execute(statement)

    def load_data(self, path, overwrite=False, partition=None):
        """
        Wraps the LOAD DATA DDL statement. Loads data into an Impala table by
        physically moving data files.

        Parameters
        ----------
        path : string
        overwrite : boolean, default False
          Overwrite the existing data in the entire table or indicated
          partition
        partition : dict, optional
          If specified, the partition must already exist

        Returns
        -------
        query : ImpalaQuery
        """
        if partition is not None:
            partition_schema = self.partition_schema()
        else:
            partition_schema = None

        stmt = ddl.LoadData(
            self._qualified_name,
            path,
            partition=partition,
            partition_schema=partition_schema,
        )

        return self._execute(stmt)

    @property
    def name(self):
        return self.op().name

    def rename(self, new_name, database=None):
        """
        Rename table inside Impala. References to the old table are no longer
        valid.

        Parameters
        ----------
        new_name : string
        database : string

        Returns
        -------
        renamed : ImpalaTable
        """
        m = fully_qualified_re.match(new_name)
        if not m and database is None:
            database = self._database
        statement = RenameTable(
            self._qualified_name, new_name, new_database=database
        )
        self._client._execute(statement)

        op = self.op().change_name(statement.new_qualified_name)
        return type(self)(op)

    def _execute(self, stmt):
        return self._client._execute(stmt)

    @property
    def is_partitioned(self):
        """
        True if the table is partitioned
        """
        return self.metadata().is_partitioned

    def partition_schema(self):
        """
        For partitioned tables, return the schema (names and types) for the
        partition columns

        Returns
        -------
        partition_schema : ibis Schema
        """
        schema = self.schema()
        name_to_type = dict(zip(schema.names, schema.types))

        result = self.partitions()

        partition_fields = []
        for x in result.columns:
            if x not in name_to_type:
                break
            partition_fields.append((x, name_to_type[x]))

        pnames, ptypes = zip(*partition_fields)
        return sch.Schema(pnames, ptypes)

    def add_partition(self, spec, location=None):
        """
        Add a new table partition, creating any new directories in HDFS if
        necessary.

        Partition parameters can be set in a single DDL statement, or you can
        use alter_partition to set them after the fact.

        Returns
        -------
        None (for now)
        """
        part_schema = self.partition_schema()
        stmt = ddl.AddPartition(
            self._qualified_name, spec, part_schema, location=location
        )
        return self._execute(stmt)

    def alter(
        self,
        location=None,
        format=None,
        tbl_properties=None,
        serde_properties=None,
    ):
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
            stmt = AlterTable(self._qualified_name, **kwds)
            return self._execute(stmt)

        return self._alter_table_helper(
            _run_ddl,
            location=location,
            format=format,
            tbl_properties=tbl_properties,
            serde_properties=serde_properties,
        )

    def set_external(self, is_external=True):
        """
        Toggle EXTERNAL table property.
        """
        self.alter(tbl_properties={'EXTERNAL': is_external})

    def alter_partition(
        self,
        spec,
        location=None,
        format=None,
        tbl_properties=None,
        serde_properties=None,
    ):
        """
        Change setting and parameters of an existing partition

        Parameters
        ----------
        spec : dict or list
          The partition keys for the partition being modified
        location : string, optional
        format : string, optional
        tbl_properties : dict, optional
        serde_properties : dict, optional

        Returns
        -------
        None (for now)
        """
        part_schema = self.partition_schema()

        def _run_ddl(**kwds):
            stmt = ddl.AlterPartition(
                self._qualified_name, spec, part_schema, **kwds
            )
            return self._execute(stmt)

        return self._alter_table_helper(
            _run_ddl,
            location=location,
            format=format,
            tbl_properties=tbl_properties,
            serde_properties=serde_properties,
        )

    def _alter_table_helper(self, f, **alterations):
        results = []
        for k, v in alterations.items():
            if v is None:
                continue
            result = f(**{k: v})
            results.append(result)
        return results

    def drop_partition(self, spec):
        """
        Drop an existing table partition
        """
        part_schema = self.partition_schema()
        stmt = ddl.DropPartition(self._qualified_name, spec, part_schema)
        return self._execute(stmt)

    def partitions(self):
        """
        Return a pandas.DataFrame giving information about this table's
        partitions. Raises an exception if the table is not partitioned.

        Returns
        -------
        partitions : pandas.DataFrame
        """
        return self._client.list_partitions(self._qualified_name)

    def stats(self):
        """
        Return results of SHOW TABLE STATS as a DataFrame. If not partitioned,
        contains only one row

        Returns
        -------
        stats : pandas.DataFrame
        """
        return self._client.table_stats(self._qualified_name)

    def column_stats(self):
        """
        Return results of SHOW COLUMN STATS as a pandas DataFrame

        Returns
        -------
        column_stats : pandas.DataFrame
        """
        return self._client.column_stats(self._qualified_name)


class ImpalaClient(SQLClient):

    """
    An Ibis client interface that uses Impala
    """

    dialect = ImpalaDialect
    database_class = ImpalaDatabase
    query_class = ImpalaQuery
    table_class = ImpalaDatabaseTable
    table_expr_class = ImpalaTable

    def __init__(self, con, hdfs_client=None, **params):
        import hdfs

        self.con = con

        if isinstance(hdfs_client, hdfs.Client):
            hdfs_client = WebHDFS(hdfs_client)
        elif hdfs_client is not None and not isinstance(hdfs_client, HDFS):
            raise TypeError(hdfs_client)

        self._hdfs = hdfs_client
        self._kudu = None

        self._temp_objects = weakref.WeakSet()

        self._ensure_temp_db_exists()

    def _build_ast(self, expr, context):
        return build_ast(expr, context)

    def _get_hdfs(self):
        if self._hdfs is None:
            raise com.IbisError(
                'No HDFS connection; must pass connection '
                'using the hdfs_client argument to '
                'ibis.impala.connect'
            )
        return self._hdfs

    def _set_hdfs(self, hdfs):
        if not isinstance(hdfs, HDFS):
            raise TypeError('must be HDFS instance')
        self._hdfs = hdfs

    hdfs = property(fget=_get_hdfs, fset=_set_hdfs)

    @property
    def kudu(self):
        from .kudu_support import KuduImpalaInterface

        if self._kudu is None:
            self._kudu = KuduImpalaInterface(self)
        return self._kudu

    def close(self):
        """
        Close Impala connection and drop any temporary objects
        """
        for obj in self._temp_objects:
            try:
                obj.drop()
            except HS2Error:
                pass

        self.con.close()

    def disable_codegen(self, disabled=True):
        """
        Turn off or on LLVM codegen in Impala query execution

        Parameters
        ----------
        disabled : boolean, default True
          To disable codegen, pass with no argument or True. To enable codegen,
          pass False
        """
        self.con.disable_codegen(disabled)

    def log(self, msg):
        log(msg)

    def _fully_qualified_name(self, name, database):
        if is_fully_qualified(name):
            return name

        database = database or self.current_database
        return '{0}.`{1}`'.format(database, name)

    def list_tables(self, like=None, database=None):
        """
        List tables in the current (or indicated) database. Like the SHOW
        TABLES command in the impala-shell.

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
            statement += ' IN {0}'.format(database)
        if like:
            m = fully_qualified_re.match(like)
            if m:
                database, quoted, unquoted = m.groups()
                like = quoted or unquoted
                return self.list_tables(like=like, database=database)
            statement += " LIKE '{0}'".format(like)

        with self._execute(statement, results=True) as cur:
            result = self._get_list(cur)

        return result

    def _get_list(self, cur):
        tuples = cur.fetchall()
        return list(map(operator.itemgetter(0), tuples))

    def set_database(self, name):
        """
        Set the default database scope for client
        """
        self.con.set_database(name)

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
        return bool(self.list_databases(like=name))

    def create_database(self, name, path=None, force=False):
        """
        Create a new Impala database

        Parameters
        ----------
        name : string
          Database name
        path : string, default None
          HDFS path where to store the database data; otherwise uses Impala
          default
        """
        if path:
            # explicit mkdir ensures the user own the dir rather than impala,
            # which is easier for manual cleanup, if necessary
            self.hdfs.mkdir(path)
        statement = CreateDatabase(name, path=path, can_exist=force)
        return self._execute(statement)

    def drop_database(self, name, force=False):
        """Drop an Impala database.

        Parameters
        ----------
        name : string
          Database name
        force : bool, default False
          If False and there are any tables in this database, raises an
          IntegrityError

        """
        if not force or self.exists_database(name):
            tables = self.list_tables(database=name)
            udfs = self.list_udfs(database=name)
            udas = self.list_udas(database=name)
        else:
            tables = []
            udfs = []
            udas = []
        if force:
            for table in tables:
                self.log('Dropping {0}'.format('{0}.{1}'.format(name, table)))
                self.drop_table_or_view(table, database=name)
            for func in udfs:
                self.log(
                    'Dropping function {0}({1})'.format(func.name, func.inputs)
                )
                self.drop_udf(
                    func.name,
                    input_types=func.inputs,
                    database=name,
                    force=True,
                )
            for func in udas:
                self.log(
                    'Dropping aggregate function {0}({1})'.format(
                        func.name, func.inputs
                    )
                )
                self.drop_uda(
                    func.name,
                    input_types=func.inputs,
                    database=name,
                    force=True,
                )
        else:
            if len(tables) > 0 or len(udfs) > 0 or len(udas) > 0:
                raise com.IntegrityError(
                    'Database {0} must be empty before '
                    'being dropped, or set '
                    'force=True'.format(name)
                )
        statement = DropDatabase(name, must_exist=not force)
        return self._execute(statement)

    def list_databases(self, like=None):
        """
        List databases in the Impala cluster. Like the SHOW DATABASES command
        in the impala-shell.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'

        Returns
        -------
        databases : list of strings
        """
        statement = 'SHOW DATABASES'
        if like:
            statement += " LIKE '{0}'".format(like)

        with self._execute(statement, results=True) as cur:
            results = self._get_list(cur)

        return results

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
        query = 'DESCRIBE {}'.format(qualified_name)

        # only pull out the first two columns which are names and types
        pairs = [row[:2] for row in self.con.fetchall(query)]

        names, types = zip(*pairs)
        ibis_types = [udf.parse_type(type.lower()) for type in types]
        names = [name.lower() for name in names]

        return sch.Schema(names, ibis_types)

    @property
    def client_options(self):
        return self.con.options

    @property
    def version(self):
        with self._execute('select version()', results=True) as cur:
            raw = self._get_list(cur)[0]

        vstring = raw.split()[2]
        return parse_version(vstring)

    def get_options(self):
        """
        Return current query options for the Impala session
        """
        query = 'SET'
        return dict(row[:2] for row in self.con.fetchall(query))

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
        Create an Impala view from a table expression

        Parameters
        ----------
        name : string
        expr : ibis TableExpr
        database : string, default None
        """
        ast = self._build_ast(expr, ImpalaDialect.make_context())
        select = ast.queries[0]
        statement = CreateView(name, select, database=database)
        return self._execute(statement)

    def drop_view(self, name, database=None, force=False):
        """
        Drop an Impala view

        Parameters
        ----------
        name : string
        database : string, default None
        force : boolean, default False
          Database may throw exception if table does not exist
        """
        statement = DropView(name, database=database, must_exist=not force)
        return self._execute(statement)

    def create_table(
        self,
        table_name,
        obj=None,
        schema=None,
        database=None,
        external=False,
        force=False,
        # HDFS options
        format='parquet',
        location=None,
        partition=None,
        like_parquet=None,
    ):
        """
        Create a new table in Impala using an Ibis table expression. This is
        currently designed for tables whose data is stored in HDFS (or
        eventually other filesystems).

        Parameters
        ----------
        table_name : string
        obj : TableExpr or pandas.DataFrame, optional
          If passed, creates table from select statement results
        schema : ibis.Schema, optional
          Mutually exclusive with obj, creates an empty table with a
          particular schema
        database : string, default None (optional)
        force : boolean, default False
          Do not create table if table with indicated name already exists
        external : boolean, default False
          Create an external table; Impala will not delete the underlying data
          when the table is dropped
        format : {'parquet'}
        location : string, default None
          Specify the directory location where Impala reads and writes files
          for the table
        partition : list of strings
          Must pass a schema to use this. Cannot partition from an expression
          (create-table-as-select)
        like_parquet : string (HDFS path), optional
          Can specify in lieu of a schema

        Examples
        --------
        >>> con.create_table('new_table_name', table_expr)  # doctest: +SKIP
        """
        if like_parquet is not None:
            raise NotImplementedError

        if obj is not None:
            if isinstance(obj, pd.DataFrame):
                from .pandas_interop import write_temp_dataframe

                writer, to_insert = write_temp_dataframe(self, obj)
            else:
                to_insert = obj
            ast = self._build_ast(to_insert, ImpalaDialect.make_context())
            select = ast.queries[0]

            statement = CTAS(
                table_name,
                select,
                database=database,
                can_exist=force,
                format=format,
                external=external,
                partition=partition,
                path=location,
            )
        elif schema is not None:
            statement = CreateTableWithSchema(
                table_name,
                schema,
                database=database,
                format=format,
                can_exist=force,
                external=external,
                path=location,
                partition=partition,
            )
        else:
            raise com.IbisError('Must pass obj or schema')

        return self._execute(statement)

    def avro_file(
        self,
        hdfs_dir,
        avro_schema,
        name=None,
        database=None,
        external=True,
        persist=False,
    ):
        """
        Create a (possibly temporary) table to read a collection of Avro data.

        Parameters
        ----------
        hdfs_dir : string
          Absolute HDFS path to directory containing avro files
        avro_schema : dict
          The Avro schema for the data as a Python dict
        name : string, default None
        database : string, default None
        external : boolean, default True
        persist : boolean, default False

        Returns
        -------
        avro_table : ImpalaTable
        """
        name, database = self._get_concrete_table_path(
            name, database, persist=persist
        )

        stmt = ddl.CreateTableAvro(
            name, hdfs_dir, avro_schema, database=database, external=external
        )
        self._execute(stmt)
        return self._wrap_new_table(name, database, persist)

    def delimited_file(
        self,
        hdfs_dir,
        schema,
        name=None,
        database=None,
        delimiter=',',
        na_rep=None,
        escapechar=None,
        lineterminator=None,
        external=True,
        persist=False,
    ):
        """
        Interpret delimited text files (CSV / TSV / etc.) as an Ibis table. See
        `parquet_file` for more exposition on what happens under the hood.

        Parameters
        ----------
        hdfs_dir : string
          HDFS directory name containing delimited text files
        schema : ibis Schema
        name : string, default None
          Name for temporary or persistent table; otherwise random one
          generated
        database : string
          Database to create the (possibly temporary) table in
        delimiter : length-1 string, default ','
          Pass None if there is no delimiter
        escapechar : length-1 string
          Character used to escape special characters
        lineterminator : length-1 string
          Character used to delimit lines
        external : boolean, default True
          Create table as EXTERNAL (data will not be deleted on drop). Not that
          if persist=False and external=False, whatever data you reference will
          be deleted
        persist : boolean, default False
          If True, do not delete the table upon garbage collection of ibis
          table object

        Returns
        -------
        delimited_table : ImpalaTable
        """
        name, database = self._get_concrete_table_path(
            name, database, persist=persist
        )

        stmt = ddl.CreateTableDelimited(
            name,
            hdfs_dir,
            schema,
            database=database,
            delimiter=delimiter,
            external=external,
            na_rep=na_rep,
            lineterminator=lineterminator,
            escapechar=escapechar,
        )
        self._execute(stmt)
        return self._wrap_new_table(name, database, persist)

    def parquet_file(
        self,
        hdfs_dir,
        schema=None,
        name=None,
        database=None,
        external=True,
        like_file=None,
        like_table=None,
        persist=False,
    ):
        """
        Make indicated parquet file in HDFS available as an Ibis table.

        The table created can be optionally named and persisted, otherwise a
        unique name will be generated. Temporarily, for any non-persistent
        external table created by Ibis we will attempt to drop it when the
        underlying object is garbage collected (or the Python interpreter shuts
        down normally).

        Parameters
        ----------
        hdfs_dir : string
          Path in HDFS
        schema : ibis Schema
          If no schema provided, and neither of the like_* argument is passed,
          one will be inferred from one of the parquet files in the directory.
        like_file : string
          Absolute path to Parquet file in HDFS to use for schema
          definitions. An alternative to having to supply an explicit schema
        like_table : string
          Fully scoped and escaped string to an Impala table whose schema we
          will use for the newly created table.
        name : string, optional
          random unique name generated otherwise
        database : string, optional
          Database to create the (possibly temporary) table in
        external : boolean, default True
          If a table is external, the referenced data will not be deleted when
          the table is dropped in Impala. Otherwise (external=False) Impala
          takes ownership of the Parquet file.
        persist : boolean, default False
          Do not drop the table upon Ibis garbage collection / interpreter
          shutdown

        Returns
        -------
        parquet_table : ImpalaTable
        """
        name, database = self._get_concrete_table_path(
            name, database, persist=persist
        )

        # If no schema provided, need to find some absolute path to a file in
        # the HDFS directory
        if like_file is None and like_table is None and schema is None:
            file_name = self.hdfs._find_any_file(hdfs_dir)
            like_file = pjoin(hdfs_dir, file_name)

        stmt = ddl.CreateTableParquet(
            name,
            hdfs_dir,
            schema=schema,
            database=database,
            example_file=like_file,
            example_table=like_table,
            external=external,
            can_exist=False,
        )
        self._execute(stmt)
        return self._wrap_new_table(name, database, persist)

    def _get_concrete_table_path(self, name, database, persist=False):
        if not persist:
            if name is None:
                name = '__ibis_tmp_{0}'.format(util.guid())

            if database is None:
                self._ensure_temp_db_exists()
                database = options.impala.temp_db
            return name, database
        else:
            if name is None:
                raise com.IbisError('Must pass table name if persist=True')
            return name, database

    def _ensure_temp_db_exists(self):
        # TODO: session memoize to avoid unnecessary `SHOW DATABASES` calls
        name, path = options.impala.temp_db, options.impala.temp_hdfs_path
        if not self.exists_database(name):
            if self._hdfs is None:
                print(
                    'Without an HDFS connection, certain functionality'
                    ' may be disabled'
                )
            else:
                self.create_database(name, path=path, force=True)

    def _wrap_new_table(self, name, database, persist):
        qualified_name = self._fully_qualified_name(name, database)

        if persist:
            t = self.table(qualified_name)
        else:
            schema = self._get_table_schema(qualified_name)
            node = ImpalaTemporaryTable(qualified_name, schema, self)
            t = self.table_expr_class(node)

        # Compute number of rows in table for better default query planning
        cardinality = t.count().execute()
        set_card = (
            "alter table {0} set tblproperties('numRows'='{1}', "
            "'STATS_GENERATED_VIA_STATS_TASK' = 'true')".format(
                qualified_name, cardinality
            )
        )
        self._execute(set_card)

        self._temp_objects.add(t)
        return t

    def text_file(self, hdfs_path, column_name='value'):
        """
        Interpret text data as a table with a single string column.

        Parameters
        ----------

        Returns
        -------
        text_table : TableExpr
        """
        pass

    def insert(
        self,
        table_name,
        obj=None,
        database=None,
        overwrite=False,
        partition=None,
        values=None,
        validate=True,
    ):
        """
        Insert into existing table.

        See ImpalaTable.insert for other parameters.

        Parameters
        ----------
        table_name : string
        database : string, default None

        Examples
        --------
        >>> table = 'my_table'
        >>> con.insert(table, table_expr)  # doctest: +SKIP

        # Completely overwrite contents
        >>> con.insert(table, table_expr, overwrite=True)  # doctest: +SKIP
        """
        table = self.table(table_name, database=database)
        return table.insert(
            obj=obj,
            overwrite=overwrite,
            partition=partition,
            values=values,
            validate=validate,
        )

    def load_data(
        self, table_name, path, database=None, overwrite=False, partition=None,
    ):
        """
        Wraps the LOAD DATA DDL statement. Loads data into an Impala table by
        physically moving data files.

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        """
        table = self.table(table_name, database=database)
        return table.load_data(path, overwrite=overwrite, partition=partition)

    def drop_table(self, table_name, database=None, force=False):
        """
        Drop an Impala table

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        force : boolean, default False
          Database may throw exception if table does not exist

        Examples
        --------
        >>> table = 'my_table'
        >>> db = 'operations'
        >>> con.drop_table(table, database=db, force=True)  # doctest: +SKIP
        """
        statement = DropTable(
            table_name, database=database, must_exist=not force
        )
        self._execute(statement)

    def truncate_table(self, table_name, database=None):
        """
        Delete all rows from, but do not drop, an existing table

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        """
        statement = TruncateTable(table_name, database=database)
        self._execute(statement)

    def drop_table_or_view(self, name, database=None, force=False):
        """
        Attempt to drop a relation that may be a view or table
        """
        try:
            self.drop_table(name, database=database)
        except Exception as e:
            try:
                self.drop_view(name, database=database)
            except Exception:
                raise e

    def cache_table(self, table_name, database=None, pool='default'):
        """
        Caches a table in cluster memory in the given pool.

        Parameters
        ----------
        table_name : string
        database : string default None (optional)
        pool : string, default 'default'
           The name of the pool in which to cache the table

        Examples
        --------
        >>> table = 'my_table'
        >>> db = 'operations'
        >>> pool = 'op_4GB_pool'
        >>> con.cache_table('my_table', database=db, pool=pool)  # noqa: E501 # doctest: +SKIP
        """
        statement = ddl.CacheTable(table_name, database=database, pool=pool)
        self._execute(statement)

    def _get_table_schema(self, tname):
        return self.get_schema(tname)

    def _get_schema_using_query(self, query):
        with self._execute(query, results=True) as cur:
            # resets the state of the cursor and closes operation
            cur.fetchall()
            names, ibis_types = self._adapt_types(cur.description)

        # per #321; most Impala tables will be lower case already, but Avro
        # data, depending on the version of Impala, might have field names in
        # the metastore cased according to the explicit case in the declared
        # avro schema. This is very annoying, so it's easier to just conform on
        # all lowercase fields from Impala.
        names = [x.lower() for x in names]

        return sch.Schema(names, ibis_types)

    def create_function(self, func, name=None, database=None):
        """
        Creates a function within Impala

        Parameters
        ----------
        func : ImpalaUDF or ImpalaUDA
          Created with wrap_udf or wrap_uda
        name : string (optional)
        database : string (optional)
        """
        if name is None:
            name = func.name
        database = database or self.current_database

        if isinstance(func, udf.ImpalaUDF):
            stmt = ddl.CreateUDF(func, name=name, database=database)
        elif isinstance(func, udf.ImpalaUDA):
            stmt = ddl.CreateUDA(func, name=name, database=database)
        else:
            raise TypeError(func)
        self._execute(stmt)

    def drop_udf(
        self,
        name,
        input_types=None,
        database=None,
        force=False,
        aggregate=False,
    ):
        """
        Drops a UDF
        If only name is given, this will search
        for the relevant UDF and drop it.
        To delete an overloaded UDF, give only a name and force=True

        Parameters
        ----------
        name : string
        input_types : list of strings (optional)
        force : boolean, default False Must be set to true to
                drop overloaded UDFs
        database : string, default None
        aggregate : boolean, default False
        """
        if not input_types:
            if not database:
                database = self.current_database
            result = self.list_udfs(database=database, like=name)
            if len(result) > 1:
                if force:
                    for func in result:
                        self._drop_single_function(
                            func.name,
                            func.inputs,
                            database=database,
                            aggregate=aggregate,
                        )
                    return
                else:
                    raise Exception(
                        "More than one function "
                        + "with {0} found.".format(name)
                        + "Please specify force=True"
                    )
            elif len(result) == 1:
                func = result.pop()
                self._drop_single_function(
                    func.name,
                    func.inputs,
                    database=database,
                    aggregate=aggregate,
                )
                return
            else:
                raise Exception("No function found with name {0}".format(name))
        self._drop_single_function(
            name, input_types, database=database, aggregate=aggregate
        )

    def drop_uda(self, name, input_types=None, database=None, force=False):
        """
        Drop aggregate function. See drop_udf for more information on the
        parameters.
        """
        return self.drop_udf(
            name, input_types=input_types, database=database, force=force
        )

    def _drop_single_function(
        self, name, input_types, database=None, aggregate=False
    ):
        stmt = ddl.DropFunction(
            name,
            input_types,
            must_exist=False,
            aggregate=aggregate,
            database=database,
        )
        self._execute(stmt)

    def _drop_all_functions(self, database):
        udfs = self.list_udfs(database=database)
        for fnct in udfs:
            stmt = ddl.DropFunction(
                fnct.name,
                fnct.inputs,
                must_exist=False,
                aggregate=False,
                database=database,
            )
            self._execute(stmt)
        udafs = self.list_udas(database=database)
        for udaf in udafs:
            stmt = ddl.DropFunction(
                udaf.name,
                udaf.inputs,
                must_exist=False,
                aggregate=True,
                database=database,
            )
            self._execute(stmt)

    def list_udfs(self, database=None, like=None):
        """
        Lists all UDFs associated with given database

        Parameters
        ----------
        database : string
        like : string for searching (optional)
        """
        if not database:
            database = self.current_database
        statement = ddl.ListFunction(database, like=like, aggregate=False)
        with self._execute(statement, results=True) as cur:
            result = self._get_udfs(cur, udf.ImpalaUDF)
        return result

    def list_udas(self, database=None, like=None):
        """
        Lists all UDAFs associated with a given database

        Parameters
        ----------
        database : string
        like : string for searching (optional)
        """
        if not database:
            database = self.current_database
        statement = ddl.ListFunction(database, like=like, aggregate=True)
        with self._execute(statement, results=True) as cur:
            result = self._get_udfs(cur, udf.ImpalaUDA)

        return result

    def _get_udfs(self, cur, klass):
        def _to_type(x):
            ibis_type = udf._impala_type_to_ibis(x.lower())
            return dt.dtype(ibis_type)

        tuples = cur.fetchall()
        if len(tuples) > 0:
            result = []
            for tup in tuples:
                out_type, sig = tup[:2]
                name, types = _split_signature(sig)
                types = _type_parser(types).types

                inputs = []
                for arg in types:
                    argm = _arg_type.match(arg)
                    var, simple = argm.groups()
                    if simple:
                        t = _to_type(simple)
                        inputs.append(t)
                    else:
                        t = _to_type(var)
                        inputs = rlz.listof(t)
                        # TODO
                        # inputs.append(varargs(t))
                        break

                output = udf._impala_type_to_ibis(out_type.lower())
                result.append(klass(inputs, output, name=name))
            return result
        else:
            return []

    def exists_udf(self, name, database=None):
        """
        Checks if a given UDF exists within a specified database

        Parameters
        ----------
        name : string, UDF name
        database : string, database name

        Returns
        -------
        if_exists : boolean
        """
        return len(self.list_udfs(database=database, like=name)) > 0

    def exists_uda(self, name, database=None):
        """
        Checks if a given UDAF exists within a specified database

        Parameters
        ----------
        name : string, UDAF name
        database : string, database name

        Returns
        -------
        if_exists : boolean
        """
        return len(self.list_udas(database=database, like=name)) > 0

    def compute_stats(self, name, database=None, incremental=False):
        """
        Issue COMPUTE STATS command for a given table

        Parameters
        ----------
        name : string
          Can be fully qualified (with database name)
        database : string, optional
        incremental : boolean, default False
          If True, issue COMPUTE INCREMENTAL STATS
        """
        maybe_inc = 'INCREMENTAL ' if incremental else ''
        cmd = 'COMPUTE {0}STATS'.format(maybe_inc)

        stmt = self._table_command(cmd, name, database=database)
        self._execute(stmt)

    def invalidate_metadata(self, name=None, database=None):
        """
        Issue INVALIDATE METADATA command, optionally only applying to a
        particular table. See Impala documentation.

        Parameters
        ----------
        name : string, optional
          Table name. Can be fully qualified (with database)
        database : string, optional
        """
        stmt = 'INVALIDATE METADATA'
        if name is not None:
            stmt = self._table_command(stmt, name, database=database)
        self._execute(stmt)

    def refresh(self, name, database=None):
        """
        Reload HDFS block location metadata for a table, for example after
        ingesting data as part of an ETL pipeline. Related to INVALIDATE
        METADATA. See Impala documentation for more.

        Parameters
        ----------
        name : string
          Table name. Can be fully qualified (with database)
        database : string, optional
        """
        # TODO(wesm): can this statement be cancelled?
        stmt = self._table_command('REFRESH', name, database=database)
        self._execute(stmt)

    def describe_formatted(self, name, database=None):
        """
        Retrieve results of DESCRIBE FORMATTED command. See Impala
        documentation for more.

        Parameters
        ----------
        name : string
          Table name. Can be fully qualified (with database)
        database : string, optional
        """
        from .metadata import parse_metadata

        stmt = self._table_command(
            'DESCRIBE FORMATTED', name, database=database
        )
        query = ImpalaQuery(self, stmt)
        result = query.execute()

        # Leave formatting to pandas
        for c in result.columns:
            result[c] = result[c].str.strip()

        return parse_metadata(result)

    def show_files(self, name, database=None):
        """
        Retrieve results of SHOW FILES command for a table. See Impala
        documentation for more.

        Parameters
        ----------
        name : string
          Table name. Can be fully qualified (with database)
        database : string, optional
        """
        stmt = self._table_command('SHOW FILES IN', name, database=database)
        return self._exec_statement(stmt)

    def list_partitions(self, name, database=None):
        stmt = self._table_command('SHOW PARTITIONS', name, database=database)
        return self._exec_statement(stmt)

    def table_stats(self, name, database=None):
        """
        Return results of SHOW TABLE STATS for indicated table. See also
        ImpalaTable.stats
        """
        stmt = self._table_command('SHOW TABLE STATS', name, database=database)
        return self._exec_statement(stmt)

    def column_stats(self, name, database=None):
        """
        Return results of SHOW COLUMN STATS for indicated table. See also
        ImpalaTable.column_stats
        """
        stmt = self._table_command(
            'SHOW COLUMN STATS', name, database=database
        )
        return self._exec_statement(stmt)

    def _exec_statement(self, stmt, adapter=None):
        query = ImpalaQuery(self, stmt)
        result = query.execute()
        if adapter is not None:
            result = adapter(result)
        return result

    def _table_command(self, cmd, name, database=None):
        qualified_name = self._fully_qualified_name(name, database)
        return '{0} {1}'.format(cmd, qualified_name)

    def _adapt_types(self, descr):
        names = []
        adapted_types = []
        for col in descr:
            names.append(col[0])
            impala_typename = col[1]
            typename = udf._impala_to_ibis_type[impala_typename.lower()]

            if typename == 'decimal':
                precision, scale = col[4:6]
                adapted_types.append(dt.Decimal(precision, scale))
            else:
                adapted_types.append(typename)
        return names, adapted_types

    def write_dataframe(self, df, path, format='csv'):
        """
        Write a pandas DataFrame to indicated file path (default: HDFS) in the
        indicated format

        Parameters
        ----------
        df : DataFrame
        path : string
          Absolute output path
        format : {'csv'}, default 'csv'

        Returns
        -------
        None (for now)
        """
        from .pandas_interop import DataFrameWriter

        writer = DataFrameWriter(self, df)
        return writer.write_csv(path)


# ----------------------------------------------------------------------
# ORM-ish usability layer


class ScalarFunction(DatabaseEntity):
    def drop(self):
        pass


class AggregateFunction(DatabaseEntity):
    def drop(self):
        pass


class ImpalaTemporaryTable(ops.DatabaseTable):
    def __del__(self):
        try:
            self.drop()
        except com.IbisError:
            pass

    def drop(self):
        try:
            self.source.drop_table(self.name)
        except ImpylaError:
            # database might have been dropped
            pass


def _validate_compatible(from_schema, to_schema):
    if set(from_schema.names) != set(to_schema.names):
        raise com.IbisInputError('Schemas have different names')

    for name in from_schema:
        lt = from_schema[name]
        rt = to_schema[name]
        if not lt.castable(rt):
            raise com.IbisInputError(
                'Cannot safely cast {0!r} to {1!r}'.format(lt, rt)
            )


def _split_signature(x):
    name, rest = x.split('(', 1)
    return name, rest[:-1]


_arg_type = re.compile(r'(.*)\.\.\.|([^\.]*)')


class _type_parser:

    NORMAL, IN_PAREN = 0, 1

    def __init__(self, value):
        self.value = value
        self.state = self.NORMAL
        self.buf = io.StringIO()
        self.types = []
        for c in value:
            self._step(c)
        self._push()

    def _push(self):
        val = self.buf.getvalue().strip()
        if val:
            self.types.append(val)
        self.buf = io.StringIO()

    def _step(self, c):
        if self.state == self.NORMAL:
            if c == '(':
                self.state = self.IN_PAREN
            elif c == ',':
                self._push()
                return
        elif self.state == self.IN_PAREN:
            if c == ')':
                self.state = self.NORMAL
        self.buf.write(c)
