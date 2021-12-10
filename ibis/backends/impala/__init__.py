"""Impala backend"""
import contextlib
import io
import operator
import re
import weakref
from posixpath import join as pjoin

import numpy as np
import pandas as pd

import ibis.common.exceptions as com
import ibis.config
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.util as util
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.base.sql.ddl import (
    CTAS,
    CreateDatabase,
    CreateTableWithSchema,
    CreateView,
    DropDatabase,
    DropTable,
    DropView,
    TruncateTable,
    fully_qualified_re,
    is_fully_qualified,
)
from ibis.config import options

from . import ddl, udf
from .client import ImpalaConnection, ImpalaDatabase, ImpalaTable
from .compat import HS2Error, ImpylaError
from .compiler import ImpalaCompiler
from .hdfs import HDFS, WebHDFS, hdfs_connect
from .pandas_interop import DataFrameWriter
from .udf import (  # noqa F408
    aggregate_function,
    scalar_function,
    wrap_uda,
    wrap_udf,
)

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


def _column_batches_to_dataframe(names, batches):
    cols = {}
    for name, chunks in zip(names, zip(*(b.columns for b in batches))):
        cols[name] = _chunks_to_pandas_array(chunks)
    return pd.DataFrame(cols, columns=names)


class Backend(BaseSQLBackend):
    name = 'impala'
    database_class = ImpalaDatabase
    table_expr_class = ImpalaTable
    HDFS = HDFS
    WebHDFS = WebHDFS
    compiler = ImpalaCompiler

    def hdfs_connect(self, *args, **kwargs):
        return hdfs_connect(*args, **kwargs)

    def do_connect(
        new_backend,
        host='localhost',
        port=21050,
        database='default',
        timeout=45,
        use_ssl=False,
        ca_cert=None,
        user=None,
        password=None,
        auth_mechanism='NOSASL',
        kerberos_service_name='impala',
        pool_size=8,
        hdfs_client=None,
    ):
        """Create a Impala Backend for use with Ibis.

        Parameters
        ----------
        host : str, optional
            Host name of the impalad or HiveServer2 in Hive
        port : int, optional
            Impala's HiveServer2 port
        database : str, optional
            Default database when obtaining new cursors
        timeout : int, optional
            Connection timeout in seconds when communicating with HiveServer2
        use_ssl : bool, optional
            Use SSL when connecting to HiveServer2
        ca_cert : str, optional
            Local path to 3rd party CA certificate or copy of server
            certificate for self-signed certificates. If SSL is enabled, but
            this argument is ``None``, then certificate validation is skipped.
        user : str, optional
            LDAP user to authenticate
        password : str, optional
            LDAP password to authenticate
        auth_mechanism : str, optional
            {'NOSASL' <- default, 'PLAIN', 'GSSAPI', 'LDAP'}.
            Use NOSASL for non-secured Impala connections.  Use PLAIN for
            non-secured Hive clusters.  Use LDAP for LDAP authenticated
            connections.  Use GSSAPI for Kerberos-secured clusters.
        kerberos_service_name : str, optional
            Specify particular impalad service principal.

        Examples
        --------
        >>> import ibis
        >>> import os
        >>> hdfs_host = os.environ.get('IBIS_TEST_NN_HOST', 'localhost')
        >>> hdfs_port = int(os.environ.get('IBIS_TEST_NN_PORT', 50070))
        >>> impala_host = os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')
        >>> impala_port = int(os.environ.get('IBIS_TEST_IMPALA_PORT', 21050))
        >>> hdfs = ibis.impala.hdfs_connect(host=hdfs_host, port=hdfs_port)
        >>> hdfs  # doctest: +ELLIPSIS
        <ibis.filesystems.WebHDFS object at 0x...>
        >>> client = ibis.impala.connect(
        ...     host=impala_host,
        ...     port=impala_port,
        ...     hdfs_client=hdfs,
        ... )
        >>> client  # doctest: +ELLIPSIS
        <ibis.backends.impala.Backend object at 0x...>

        Returns
        -------
        Backend
        """
        import hdfs

        new_backend._kudu = None
        new_backend._temp_objects = set()

        if hdfs_client is None or isinstance(hdfs_client, HDFS):
            new_backend._hdfs = hdfs_client
        elif isinstance(hdfs_client, hdfs.Client):
            new_backend._hdfs = WebHDFS(hdfs_client)
        else:
            raise TypeError(hdfs_client)

        params = {
            'host': host,
            'port': port,
            'database': database,
            'timeout': timeout,
            'use_ssl': use_ssl,
            'ca_cert': ca_cert,
            'user': user,
            'password': password,
            'auth_mechanism': auth_mechanism,
            'kerberos_service_name': kerberos_service_name,
        }
        new_backend.con = ImpalaConnection(pool_size=pool_size, **params)

        new_backend._ensure_temp_db_exists()

    @property
    def version(self):
        cursor = self.raw_sql('select version()')
        result = cursor.fetchone()[0]
        cursor.release()
        return result

    def register_options(self):
        ibis.config.register_option(
            'temp_db',
            '__ibis_tmp',
            'Database to use for temporary tables, views. functions, etc.',
        )
        ibis.config.register_option(
            'temp_hdfs_path',
            '/tmp/ibis',
            'HDFS path for storage of temporary data',
        )

    def list_databases(self, like=None):
        cur = self.raw_sql('SHOW DATABASES')
        databases = self._get_list(cur)
        cur.release()
        return self._filter_with_like(databases, like)

    def list_tables(self, like=None, database=None):
        statement = 'SHOW TABLES'
        if database is not None:
            statement += f' IN {database}'
        if like:
            m = fully_qualified_re.match(like)
            if m:
                database, quoted, unquoted = m.groups()
                like = quoted or unquoted
                return self.list_tables(like=like, database=database)
            statement += f" LIKE '{like}'"

        return self._filter_with_like(
            [row[0] for row in self.raw_sql(statement).fetchall()]
        )

    def fetch_from_cursor(self, cursor, schema):
        batches = cursor.fetchall(columnar=True)
        names = [x[0] for x in cursor.description]
        df = _column_batches_to_dataframe(names, batches)
        if schema:
            return schema.apply_to(df)
        return df

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
        """Close the connection and drop temporary objects."""
        while self._temp_objects:
            finalizer = self._temp_objects.pop()
            with contextlib.suppress(HS2Error):
                finalizer()

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

    def _fully_qualified_name(self, name, database):
        if is_fully_qualified(name):
            return name

        database = database or self.current_database
        return f'{database}.`{name}`'

    def _get_list(self, cur):
        tuples = cur.fetchall()
        return list(map(operator.itemgetter(0), tuples))

    @util.deprecated(version='2.0', instead='a new connection to database')
    def set_database(self, name):
        # XXX The parent `Client` has a generic method that calls this same
        # method in the backend. But for whatever reason calling this code from
        # that method doesn't seem to work. Maybe `con` is a copy?
        self.con.set_database(name)

    @property
    def current_database(self):
        # XXX The parent `Client` has a generic method that calls this same
        # method in the backend. But for whatever reason calling this code from
        # that method doesn't seem to work. Maybe `con` is a copy?
        return self.con.database

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
        return self.raw_sql(statement)

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
        if not force or name in self.list_databases():
            tables = self.list_tables(database=name)
            udfs = self.list_udfs(database=name)
            udas = self.list_udas(database=name)
        else:
            tables = []
            udfs = []
            udas = []
        if force:
            for table in tables:
                util.log('Dropping {}'.format(f'{name}.{table}'))
                self.drop_table_or_view(table, database=name)
            for func in udfs:
                util.log(f'Dropping function {func.name}({func.inputs})')
                self.drop_udf(
                    func.name,
                    input_types=func.inputs,
                    database=name,
                    force=True,
                )
            for func in udas:
                util.log(
                    'Dropping aggregate function {}({})'.format(
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
                    'Database {} must be empty before '
                    'being dropped, or set '
                    'force=True'.format(name)
                )
        statement = DropDatabase(name, must_exist=not force)
        return self.raw_sql(statement)

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
        query = f'DESCRIBE {qualified_name}'

        # only pull out the first two columns which are names and types
        pairs = [row[:2] for row in self.con.fetchall(query)]

        names, types = zip(*pairs)
        ibis_types = [udf.parse_type(type.lower()) for type in types]
        names = [name.lower() for name in names]

        return sch.Schema(names, ibis_types)

    @property
    def client_options(self):
        return self.con.options

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
            raise ValueError(f'Unknown codec: {codec}')

        self.set_options({'COMPRESSION_CODEC': codec})

    def create_view(self, name, expr, database=None):
        """
        Create an Impala view from a table expression

        Parameters
        ----------
        name : string
        expr : ibis TableExpr
        database : string, default None
        """
        ast = self.compiler.to_ast(expr)
        select = ast.queries[0]
        statement = CreateView(name, select, database=database)
        return self.raw_sql(statement)

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
        return self.raw_sql(statement)

    @contextlib.contextmanager
    def _setup_insert(self, obj):
        if isinstance(obj, pd.DataFrame):
            with DataFrameWriter(self, obj) as writer:
                yield writer.delimited_table(writer.write_temp_csv())
        else:
            yield obj

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
            with self._setup_insert(obj) as to_insert:
                ast = self.compiler.to_ast(to_insert)
                select = ast.queries[0]

                self.raw_sql(
                    CTAS(
                        table_name,
                        select,
                        database=database,
                        can_exist=force,
                        format=format,
                        external=external,
                        partition=partition,
                        path=location,
                    )
                )
        elif schema is not None:
            self.raw_sql(
                CreateTableWithSchema(
                    table_name,
                    schema,
                    database=database,
                    format=format,
                    can_exist=force,
                    external=external,
                    path=location,
                    partition=partition,
                )
            )
        else:
            raise com.IbisError('Must pass obj or schema')

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
        self.raw_sql(stmt)
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
        self.raw_sql(stmt)
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
        self.raw_sql(stmt)
        return self._wrap_new_table(name, database, persist)

    def _get_concrete_table_path(self, name, database, persist=False):
        if not persist:
            if name is None:
                name = f'__ibis_tmp_{util.guid()}'

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
        if name not in self.list_databases():
            if self._hdfs is None:
                print(
                    'Without an HDFS connection, certain functionality'
                    ' may be disabled'
                )
            else:
                self.create_database(name, path=path, force=True)

    def _drop_table(self, name: str) -> None:
        # database might have been dropped, so we suppress the
        # corresponding Exception
        with contextlib.suppress(ImpylaError):
            self.drop_table(name)

    def _wrap_new_table(self, name, database, persist):
        qualified_name = self._fully_qualified_name(name, database)
        t = self.table(qualified_name)
        if not persist:
            self._temp_objects.add(
                weakref.finalize(t, self._drop_table, qualified_name)
            )

        # Compute number of rows in table for better default query planning
        cardinality = t.count().execute()
        set_card = (
            "alter table {} set tblproperties('numRows'='{}', "
            "'STATS_GENERATED_VIA_STATS_TASK' = 'true')".format(
                qualified_name, cardinality
            )
        )
        self.raw_sql(set_card)

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
        self,
        table_name,
        path,
        database=None,
        overwrite=False,
        partition=None,
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
        self.raw_sql(statement)

    def truncate_table(self, table_name, database=None):
        """
        Delete all rows from, but do not drop, an existing table

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        """
        statement = TruncateTable(table_name, database=database)
        self.raw_sql(statement)

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
        self.raw_sql(statement)

    def _get_schema_using_query(self, query):
        cur = self.raw_sql(query)
        # resets the state of the cursor and closes operation
        cur.fetchall()
        names, ibis_types = self._adapt_types(cur.description)
        cur.release()

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
        self.raw_sql(stmt)

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
                        + f"with {name} found."
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
                raise Exception(f"No function found with name {name}")
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
        self.raw_sql(stmt)

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
            self.raw_sql(stmt)
        udafs = self.list_udas(database=database)
        for udaf in udafs:
            stmt = ddl.DropFunction(
                udaf.name,
                udaf.inputs,
                must_exist=False,
                aggregate=True,
                database=database,
            )
            self.raw_sql(stmt)

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
        cur = self.raw_sql(statement)
        result = self._get_udfs(cur, udf.ImpalaUDF)
        cur.release()
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
        cur = self.raw_sql(statement)
        result = self._get_udfs(cur, udf.ImpalaUDA)
        cur.release()

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
        cmd = f'COMPUTE {maybe_inc}STATS'

        stmt = self._table_command(cmd, name, database=database)
        self.raw_sql(stmt)

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
        self.raw_sql(stmt)

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
        self.raw_sql(stmt)

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
        result = self._exec_statement(stmt)

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

    def _exec_statement(self, stmt):
        return self.fetch_from_cursor(
            self.raw_sql(stmt, results=True), schema=None
        )

    def _table_command(self, cmd, name, database=None):
        qualified_name = self._fully_qualified_name(name, database)
        return f'{cmd} {qualified_name}'

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
        writer = DataFrameWriter(self, df)
        return writer.write_csv(path)
