import threading
import time
import traceback
import weakref
from collections import deque

import ibis.common.exceptions as com
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base import Database
from ibis.backends.base.sql.compiler import DDL, DML
from ibis.backends.base.sql.ddl import (
    AlterTable,
    InsertSelect,
    RenameTable,
    fully_qualified_re,
)

from . import ddl
from .compat import HS2Error, impyla


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
        return self.client.list_udfs(like=like, database=self.name)

    def list_udas(self, like=None):
        return self.client.list_udas(like=like, database=self.name)


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
        with self.lock:
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
        util.log(query)

        try:
            cursor.execute(query)
        except Exception:
            cursor.release()
            util.log(
                'Exception caused by {}: {}'.format(
                    query, traceback.format_exc()
                )
            )
            raise

        return cursor

    def fetchall(self, query):
        with self.execute(query) as cur:
            results = cur.fetchall()
        return results

    def _get_cursor(self):
        try:
            cursor = self.connection_pool.popleft()
        except IndexError:  # deque is empty
            with self.lock:
                # NB: Do not put a lock around the entire if statement.
                # This will cause a deadlock because _new_cursor calls the
                # ImpalaCursor constructor which takes a lock to increment the
                # connection pool size.
                connection_pool_size = self.connection_pool_size
            if connection_pool_size < self.max_pool_size:
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
        with self.con.lock:
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
            query = f'SET {k} = {v!r}'
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


class ImpalaTable(ir.TableExpr):

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
                'Cannot determine database name from {}'.format(
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
        if values is not None:
            raise NotImplementedError

        with self._client._setup_insert(obj) as expr:
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

            ast = self._client.compiler.to_ast(expr)
            select = ast.queries[0]
            statement = InsertSelect(
                self._qualified_name,
                select,
                partition=partition,
                partition_schema=partition_schema,
                overwrite=overwrite,
            )
            return self._client.raw_sql(statement.compile())

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

        return self._client.raw_sql(stmt.compile())

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
        self._client.raw_sql(statement)

        op = self.op().change_name(statement.new_qualified_name)
        return type(self)(op)

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
        return self._client.raw_sql(stmt)

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
            return self._client.raw_sql(stmt)

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
            return self._client.raw_sql(stmt)

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
        return self._client.raw_sql(stmt)

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


# ----------------------------------------------------------------------
# ORM-ish usability layer


class ScalarFunction:
    def drop(self):
        pass


class AggregateFunction:
    def drop(self):
        pass


def _validate_compatible(from_schema, to_schema):
    if set(from_schema.names) != set(to_schema.names):
        raise com.IbisInputError('Schemas have different names')

    for name in from_schema:
        lt = from_schema[name]
        rt = to_schema[name]
        if not lt.castable(rt):
            raise com.IbisInputError(f'Cannot safely cast {lt!r} to {rt!r}')
