from ibis.compat import parse_version
from ibis.client import Database, Query, SQLClient, DatabaseEntity
from ibis.mapd.compiler import MapDDialect, build_ast
from ibis.mapd import ddl
from ibis.sql.compiler import DDL, DML
from ibis.util import log
from pymapd.cursor import Cursor

try:
    from pygdf.dataframe import DataFrame as GPUDataFrame
except ImportError:
    GPUDataFrame = None

import regex as re
import pandas as pd
import pymapd

import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir

EXECUTION_TYPE_ICP = 1
EXECUTION_TYPE_ICP_GPU = 2
EXECUTION_TYPE_CURSOR = 3

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


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
    return


class MapDDataType(object):

    __slots__ = 'typename', 'nullable'

    # using impala.client._HS2_TTypeId_to_dtype as reference
    # https://www.mapd.com/docs/latest/mapd-core-guide/fixed-encoding/
    dtypes = {
        'BIGINT': dt.int64,
        'BOOLEAN': dt.Boolean,
        'BOOL': dt.Boolean,
        'CHAR': dt.string,
        'DATE': dt.date,
        'DECIMAL': dt.float64,
        'DOUBLE': dt.float64,
        'INT': dt.int32,
        'INTEGER': dt.int32,
        'FLOAT': dt.float32,
        'NULL': dt.Null,
        'NUMERIC': dt.float64,
        'REAL': dt.float32,
        'SMALLINT': dt.int16,
        'STR': dt.string,
        'TEXT': dt.string,
        'TIME': dt.time,
        'TIMESTAMP': dt.timestamp,
        'VARCHAR': dt.string,
    }

    ibis_dtypes = {
        v: k for k, v in dtypes.items()
    }

    def __init__(self, typename, nullable=False):
        if typename not in self.dtypes:
            raise com.UnsupportedBackendType(typename)
        self.typename = typename
        self.nullable = nullable

    def __str__(self):
        if self.nullable:
            return 'Nullable({})'.format(self.typename)
        else:
            return self.typename

    def __repr__(self):
        return '<MapD {}>'.format(str(self))

    @classmethod
    def parse(cls, spec):
        if spec.startswith('Nullable'):
            return cls(spec[9:-1], nullable=True)
        else:
            return cls(spec)

    def to_ibis(self):
        return self.dtypes[self.typename](nullable=self.nullable)

    @classmethod
    def from_ibis(cls, dtype, nullable=None):
        dtype_ = type(dtype)
        if dtype_ in cls.ibis_dtypes:
            typename = cls.ibis_dtypes[dtype_]
        elif dtype in cls.ibis_dtypes:
            typename = cls.ibis_dtypes[dtype]
        else:
            raise NotImplemented('{0} not Implemented'.format(dtype))

        if nullable is None:
            nullable = dtype.nullable
        return cls(typename, nullable=nullable)


class MapDCursor(object):
    """Cursor to allow the MapD client to reuse machinery in ibis/client.py
    """

    def __init__(self, cursor):
        self.cursor = cursor

    def to_df(self):
        if isinstance(self.cursor, Cursor):
            col_names = [c.name for c in self.cursor.description]
            result = pd.DataFrame(self.cursor.fetchall(), columns=col_names)
        elif self.cursor is None:
            result = pd.DataFrame([])
        else:
            result = self.cursor

        return result

    def __enter__(self):
        # For compatibility when constructed from Query.execute()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MapDQuery(Query):
    """

    """
    def _fetch(self, cursor):
        # check if cursor is a pymapd cursor.Cursor
        return self.schema().apply_to(cursor.to_df())


class MapDTable(ir.TableExpr, DatabaseEntity):

    """
    References a physical table in the MapD metastore
    """

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

    def drop(self):
        """
        Drop the table from the database
        """
        self._client.drop_table_or_view(self._qualified_name)

    def truncate(self):
        self._client.truncate_table(self._qualified_name)

    def insert(self, obj=None, values=None, validate=True):
        """
        Insert into Impala table. Wraps ImpalaClient.insert

        Parameters
        ----------
        obj : TableExpr or pandas DataFrame
        values: , Default None
        validate : boolean, default True
          If True, do more rigorous validation that schema of table being
          inserted is compatible with the existing table

        Examples
        --------
        >>> t.insert(table_expr)  # doctest: +SKIP
        """
        if values is not None:
            raise NotImplementedError

        if isinstance(obj, pd.DataFrame):
            from ibis.mapd.pandas_interop import write_temp_dataframe
            writer, expr = write_temp_dataframe(self._client, obj)
        else:
            expr = obj


        if validate:
            existing_schema = self.schema()
            insert_schema = expr.schema()
            if not insert_schema.equals(existing_schema):
                _validate_compatible(insert_schema, existing_schema)

        ast = build_ast(expr, MapDDialect.make_context())
        select = ast.queries[0]
        statement = ddl.InsertSelect(self._unqualified_name, select)
        return self._execute(statement)

    def load_data(self, path, overwrite=False):
        """
        Wraps the LOAD DATA DDL statement. Loads data into an MapD table by
        physically moving data files.

        Parameters
        ----------
        path : string
        overwrite : boolean, default False
          Overwrite the existing data in the entire table or indicated
          partition

        Returns
        -------
        query : MapDQuery
        """
        stmt = ddl.LoadData(self._qualified_name, path)
        return self._execute(stmt)

    @property
    def name(self):
        return self.op().name

    def rename(self, new_name, database=None):
        """
        Rename table inside MapD. References to the old table are no longer
        valid.

        Parameters
        ----------
        new_name : string
        database : string

        Returns
        -------
        renamed : MapDTable
        """
        m = ddl.fully_qualified_re.match(new_name)
        if not m and database is None:
            database = self._database

        statement = ddl.RenameTable(
            self._qualified_name, new_name, new_database=database
        )

        self._client._execute(statement)

        op = self.op().change_name(statement.new_qualified_name)
        return type(self)(op)

    def _execute(self, stmt):
        return self._client._execute(stmt)

    def alter(self, tbl_properties=None):
        """
        Change setting and parameters of the table.

        Parameters
        ----------
        tbl_properties : dict, optional

        Returns
        -------
        None (for now)
        """
        def _run_ddl(**kwds):
            stmt = ddl.AlterTable(self._qualified_name, **kwds)
            return self._execute(stmt)

        return self._alter_table_helper(
            _run_ddl, tbl_properties=tbl_properties
        )

    def _alter_table_helper(self, f, **alterations):
        results = []
        for k, v in alterations.items():
            if v is None:
                continue
            result = f(**{k: v})
            results.append(result)
        return results


class MapDClient(SQLClient):
    """

    """
    database_class = Database
    sync_query = MapDQuery
    dialect = MapDDialect
    table_expr_class = MapDTable

    def __init__(
        self, uri=None, user=None, password=None,
        host=None, port=9091, database=None,
        protocol='binary', execution_type=EXECUTION_TYPE_CURSOR
    ):
        """

        Parameters
        ----------
        uri : str
        user : str
        password : str
        host : str
        port : int
        database : str
        protocol : {'binary', 'http', 'https'}
        execution_type : {
          EXECUTION_TYPE_ICP, EXECUTION_TYPE_ICP_GPU, EXECUTION_TYPE_CURSOR
        }

        """
        self.uri = uri
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = database
        self.protocol = protocol

        if execution_type not in (
            EXECUTION_TYPE_ICP,
            EXECUTION_TYPE_ICP_GPU,
            EXECUTION_TYPE_CURSOR,
        ):
            raise Exception('Execution type defined not available.')

        self.execution_type = execution_type

        self.con = pymapd.connect(
            uri=uri, user=user, password=password, host=host,
            port=port, dbname=database, protocol=protocol
        )

    def log(self, msg):
        log(msg)

    def close(self):
        """Close MapD connection and drop any temporary objects"""
        self.con.close()

    def _build_ast(self, expr, context):
        """
        Required.

        expr:
        context:
        :return:
        """
        result = build_ast(expr, context)
        return result

    def _fully_qualified_name(self, name, database):
        if fully_qualified_re.search(name):
            return name

        database = database or self.current_database
        return '{}.{}'.format(database, name)

    def _get_table_schema(self, table_name, database=None):
        """

        Parameters
        ----------
        table_name : str
        database : str

        Returns
        -------
        schema : ibis Schema

        """
        table_name_ = table_name.split('.')
        if len(table_name_) == 2:
            database, table_name = table_name_
        return self.get_schema(table_name, database)

    def _execute(self, query, results=True):
        """

        query:
        :return:
        """
        if isinstance(query, (DDL, DML)):
            query = query.compile()

        if self.execution_type == EXECUTION_TYPE_ICP:
            execute = self.con.select_ipc
        elif self.execution_type == EXECUTION_TYPE_ICP_GPU:
            execute = self.con.select_ipc_gpu
        else:
            execute = self.con.cursor().execute

        try:
            result = MapDCursor(execute(query))
        except Exception as e:
            raise Exception('{}: {}'.format(e, query))

        if results:
            return result

    def create_table(
        self, table_name, obj=None, schema=None, database=None, force=False
    ):
        """
        Create a new table in MapD using an Ibis table expression.

        Parameters
        ----------
        table_name : string
        obj : TableExpr or pandas.DataFrame, optional
          If passed, creates table from select statement results
        schema : ibis.Schema, optional
          Mutually exclusive with expr, creates an empty table with a
          particular schema
        database : string, default None (optional)
        force : boolean, default False
          Do not create table if table with indicated name already exists

        Examples
        --------
        >>> con.create_table('new_table_name', table_expr)  # doctest: +SKIP
        """

        if obj is not None:
            if isinstance(obj, pd.DataFrame):
                from ibis.mapd.pandas_interop import write_temp_dataframe
                writer, to_insert = write_temp_dataframe(self._client, obj)
            else:
                to_insert = obj
            ast = self._build_ast(to_insert, MapDDialect.make_context())
            select = ast.queries[0]

            statement = ddl.CTAS(
                table_name, select,
                database=database,
                can_exist=force,
            )
        elif schema is not None:
            statement = ddl.CreateTableWithSchema(
                table_name, schema,
                database=database,
                can_exist=force,
            )
        else:
            raise com.IbisError('Must pass expr or schema')

        return self._execute(statement, False)

    def delimited_file(
        self, buf, schema, name=None, database=None, delimiter=',',
        na_rep=None, escapechar=None, lineterminator=None, persist=False
    ):
        """
        Interpret delimited text files (CSV / TSV / etc.) as an Ibis table. See
        `parquet_file` for more exposition on what happens under the hood.

        Parameters
        ----------
        schema : ibis Schema
        name : string, default None
          Name for temporary or persistent table; otherwise random one
          generated
        buf: buffer
        database : string
          Database to create the (possibly temporary) table in
        delimiter : length-1 string, default ','
          Pass None if there is no delimiter
        escapechar : length-1 string
          Character used to escape special characters
        lineterminator : length-1 string
          Character used to delimit lines
        persist : boolean, default False
          If True, do not delete the table upon garbage collection of ibis
          table object

        Returns
        -------
        delimited_table : MapDTable
        """
        name = name
        database = database or self.db_name

        stmt = ddl.CreateTableDelimited(
            name, buf, schema,
            database=database,
            delimiter=delimiter,
            na_rep=na_rep,
            lineterminator=lineterminator,
            escapechar=escapechar
        )
        self._execute(stmt)
        return self._wrap_new_table(name, database, persist)

    def drop_table(self, table_name, database=None, force=False):
        """
        Drop an MapD table

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
        statement = ddl.DropTable(
            table_name, database=database, must_exist=not force
        )
        self._execute(statement, False)

    def drop_view(self, name, database=None, force=False):
        """
        Drop an MapD view

        Parameters
        ----------
        name : string
        database : string, default None
        force : boolean, default False
          Database may throw exception if table does not exist
        """
        statement = ddl.DropView(
            name, database=database, must_exist=not force
        )
        return self._execute(statement, False)

    def truncate_table(self, table_name, database=None):
        """
        Delete all rows from, but do not drop, an existing table

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        """
        statement = ddl.TruncateTable(table_name, database=database)
        self._execute(statement, False)

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

    def database(self, name=None):
        """Connect to a database called `name`.

        Parameters
        ----------
        name : str, optional
            The name of the database to connect to. If ``None``, return
            the database named ``self.current_database``.

        Returns
        -------
        db : Database
            An :class:`ibis.client.Database` instance.

        Notes
        -----
        This creates a new connection if `name` is both not ``None`` and not
        equal to the current database.
        """
        if name == self.current_database or name is None:
            return self.database_class(self.current_database, self)
        else:
            client_class = type(self)
            new_client = client_class(
                uri=self.uri, user=self.user, password=self.password,
                host=self.host, port=self.port, database=name,
                protocol=self.protocol, execution_type=self.execution_type
            )
            return self.database_class(name, new_client)

    def insert(
        self, table_name, obj=None, database=None, values=None, validate=True
    ):
        """
        Insert into existing table.

        See MapDTable.insert for other parameters.

        Parameters
        ----------
        table_name : string
        database : string, default None

        Examples
        --------
        >>> table = 'my_table'
        >>> con.insert(table, table_expr)  # doctest: +SKIP

        """
        table = self.table(table_name, database=database)
        return table.insert(
            obj=obj, values=values, validate=validate
        )

    def load_data(self, table_name, path, database=None, overwrite=False):
        """
        Wraps the LOAD DATA DDL statement. Loads data into an MapD table by
        physically moving data files.

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        """
        table = self.table(table_name, database=database)
        return table.load_data(path, overwrite=overwrite)

    @property
    def current_database(self):
        return self.db_name

    def set_database(self, name):
        raise NotImplementedError(
            'Cannot set database with MapD client. To use a different'
            ' database, use client.database({!r})'.format(name)
        )

    def exists_database(self, name):
        raise NotImplementedError()

    def list_databases(self, like=None):
        raise NotImplementedError()

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
        return bool(self.list_tables(like=name, database=database))

    def list_tables(self, like=None, database=None):
        tables = self.con.get_tables()

        if like is None:
            return tables
        return list(filter(lambda t: t == like, tables))

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
        col_names = []
        col_types = []

        for col in self.con.get_table_details(table_name):
            col_names.append(col.name)
            col_types.append(MapDDataType.parse(col.type))

        return sch.schema([
            (col.name, MapDDataType.parse(col.type))
            for col in self.con.get_table_details(table_name)
        ])

    @property
    def version(self):
        return parse_version(pymapd.__version__)


@dt.dtype.register(MapDDataType)
def mapd_to_ibis_dtype(mapd_dtype):
    """
    Register MapD Data Types

    mapd_dtype:
    :return:
    """
    return mapd_dtype.to_ibis()
