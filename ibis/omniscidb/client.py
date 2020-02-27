"""Ibis OmniSciDB Client."""
from pathlib import Path
from typing import Union

import pandas as pd
import pkg_resources
import pymapd
import regex as re
from pymapd._parsers import _extract_column_details
from pymapd.cursor import Cursor
from pymapd.dtypes import TDatumType as pymapd_dtype

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.client import Database, DatabaseEntity, Query, SQLClient
from ibis.omniscidb import ddl
from ibis.omniscidb.compiler import OmniSciDBDialect, build_ast
from ibis.sql.compiler import DDL, DML
from ibis.util import log

try:
    from cudf.dataframe.dataframe import DataFrame as GPUDataFrame
except (ImportError, OSError):
    GPUDataFrame = None

# used to check if geopandas and shapely is available
FULL_GEO_SUPPORTED = False
try:
    import geopandas
    import shapely.wkt

    FULL_GEO_SUPPORTED = True
except ImportError:
    ...

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


class PyMapDVersionError(Exception):
    """PyMapD version error exception."""

    pass


class OmniSciDBDataType:
    """OmniSciDB Backend Data Type."""

    __slots__ = 'typename', 'nullable'

    # using impala.client._HS2_TTypeId_to_dtype as reference
    # NOTE: any updates here should be reflected to
    #       omniscidb.operations._sql_type_names
    dtypes = {
        'BIGINT': dt.int64,
        'BOOL': dt.Boolean,
        'DATE': dt.date,
        'DECIMAL': dt.Decimal(18, 9),
        'DOUBLE': dt.double,
        'FLOAT': dt.float32,
        'INT': dt.int32,
        'LINESTRING': dt.linestring,
        'MULTIPOLYGON': dt.multipolygon,
        'NULL': dt.Null,
        'NUMERIC': dt.Decimal(18, 9),
        'POINT': dt.point,
        'POLYGON': dt.polygon,
        'SMALLINT': dt.int16,
        'STR': dt.string,
        'TIME': dt.time,
        'TIMESTAMP': dt.timestamp,
        'TINYINT': dt.int8,
    }

    ibis_dtypes = {v: k for k, v in dtypes.items()}

    # NOTE: any updates here should be reflected to
    #       omniscidb.operations._sql_type_names
    _omniscidb_to_ibis_dtypes = {
        'BIGINT': 'int64',
        'BOOLEAN': 'Boolean',
        'BOOL': 'Boolean',
        'CHAR': 'string',
        'DATE': 'date',
        'DECIMAL': 'decimal',
        'DOUBLE': 'double',
        'INT': 'int32',
        'INTEGER': 'int32',
        'FLOAT': 'float32',
        'NUMERIC': 'float64',
        'REAL': 'float32',
        'SMALLINT': 'int16',
        'STR': 'string',
        'TEXT': 'string',
        'TIME': 'time',
        'TIMESTAMP': 'timestamp',
        'TINYINT': 'int8',
        'VARCHAR': 'string',
        'POINT': 'point',
        'LINESTRING': 'linestring',
        'POLYGON': 'polygon',
        'MULTIPOLYGON': 'multipolygon',
    }

    def __init__(self, typename, nullable=True):
        if typename not in self.dtypes:
            raise com.UnsupportedBackendType(typename)
        self.typename = typename
        self.nullable = nullable

    def __str__(self):
        """Return the data type name."""
        if self.nullable:
            return 'Nullable({})'.format(self.typename)
        else:
            return self.typename

    def __repr__(self):
        """Return the backend name and the datatype name."""
        return '<OmniSciDB {}>'.format(str(self))

    @classmethod
    def parse(cls, spec: str):
        """Return a OmniSciDBDataType related to the given data type name.

        Parameters
        ----------
        spec : string

        Returns
        -------
        OmniSciDBDataType
        """
        if spec.startswith('Nullable'):
            return cls(spec[9:-1], nullable=True)
        else:
            return cls(spec)

    def to_ibis(self):
        """
        Return the Ibis data type correspondent to the current OmniSciDB type.

        Returns
        -------
        ibis.expr.datatypes.DataType
        """
        return self.dtypes[self.typename](nullable=self.nullable)

    @classmethod
    def from_ibis(cls, dtype, nullable=None):
        """
        Return a OmniSciDBDataType correspondent to the given Ibis data type.

        Parameters
        ----------
        dtype : ibis.expr.datatypes.DataType
        nullable : bool

        Returns
        -------
        OmniSciDBDataType

        Raises
        ------
        NotImplementedError
            if the given data type was not implemented.
        """
        dtype_ = type(dtype)
        if dtype_ in cls.ibis_dtypes:
            typename = cls.ibis_dtypes[dtype_]
        elif dtype in cls.ibis_dtypes:
            typename = cls.ibis_dtypes[dtype]
        else:
            raise NotImplementedError('{} dtype not implemented'.format(dtype))

        if nullable is None:
            nullable = dtype.nullable
        return cls(typename, nullable=nullable)


class OmniSciDBDefaultCursor:
    """Default cursor that exports a result to Pandas Data Frame."""

    def __init__(self, cursor):
        self.cursor = cursor

    def to_df(self):
        """Convert the cursor to a data frame.

        Returns
        -------
        dataframe : pandas.DataFrame
        """
        if isinstance(self.cursor, Cursor):
            col_names = [c.name for c in self.cursor.description]
            result = pd.DataFrame(self.cursor.fetchall(), columns=col_names)
        elif self.cursor is None:
            result = pd.DataFrame([])
        else:
            result = self.cursor

        return result

    def __enter__(self):
        """For compatibility when constructed from Query.execute()."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit when using `with` statement."""
        pass


class OmniSciDBGeoCursor(OmniSciDBDefaultCursor):
    """Cursor that exports result to GeoPandas Data Frame."""

    def to_df(self):
        """Convert the cursor to a data frame.

        Returns
        -------
        dataframe : pandas.DataFrame
        """
        cursor = self.cursor
        cursor_description = cursor.description

        if not isinstance(cursor, Cursor):
            if cursor is None:
                return geopandas.GeoDataFrame([])
            return cursor

        col_names = [c.name for c in cursor_description]
        result = pd.DataFrame(cursor.fetchall(), columns=col_names)

        # get geo types from pymapd
        geotypes = (
            pymapd_dtype.POINT,
            pymapd_dtype.LINESTRING,
            pymapd_dtype.POLYGON,
            pymapd_dtype.MULTIPOLYGON,
            pymapd_dtype.GEOMETRY,
            pymapd_dtype.GEOGRAPHY,
        )

        geo_column = None

        for d in cursor_description:
            field_name = d.name
            if d.type_code in geotypes:
                # use the first geo column found as default geometry
                # geopandas doesn't allow multiple GeoSeries
                # to specify other column as a geometry on a GeoDataFrame
                # use something like: df.set_geometry('buffers').plot()
                geo_column = geo_column or field_name
                result[field_name] = result[field_name].apply(
                    shapely.wkt.loads
                )
        if geo_column:
            result = geopandas.GeoDataFrame(result, geometry=geo_column)
        return result


class OmniSciDBQuery(Query):
    """OmniSciDB Query class."""

    def _fetch(self, cursor):
        # check if cursor is a pymapd cursor.Cursor
        return self.schema().apply_to(cursor.to_df())


class OmniSciDBTable(ir.TableExpr, DatabaseEntity):
    """References a physical table in the OmniSciDB metastore."""

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

    @com.mark_as_unsupported
    def invalidate_metadata(self):
        """Invalidate table metadata.

        Raises
        ------
        common.exceptions.UnsupportedOperationError
        """

    @com.mark_as_unsupported
    def refresh(self):
        """Refresh table metadata.

        Raises
        ------
        common.exceptions.UnsupportedOperationError
        """

    def metadata(self):
        """
        Return parsed results of DESCRIBE FORMATTED statement.

        Returns
        -------
        metadata : pandas.DataFrame
        """
        return pd.DataFrame(
            [
                (
                    col.name,
                    OmniSciDBDataType.parse(col.type),
                    col.precision,
                    col.scale,
                    col.comp_param,
                    col.encoding,
                )
                for col in self._client.con.get_table_details(
                    self._qualified_name
                )
            ],
            columns=[
                'column_name',
                'type',
                'precision',
                'scale',
                'comp_param',
                'encoding',
            ],
        )

    describe_formatted = metadata

    def drop(self):
        """Drop the table from the database."""
        self._client.drop_table_or_view(self._qualified_name)

    def truncate(self):
        """Delete all rows from, but do not drop, an existing table."""
        self._client.truncate_table(self._qualified_name)

    def load_data(self, df):
        """
        Load a data frame into database.

        Wraps the LOAD DATA DDL statement. Loads data into an OmniSciDB table
        from pandas.DataFrame or pyarrow.Table

        Parameters
        ----------
        df: pandas.DataFrame or pyarrow.Table

        Returns
        -------
        query : OmniSciDBQuery
        """
        stmt = ddl.LoadData(self._qualified_name, df)
        return self._execute(stmt)

    def read_csv(
        self,
        path: Union[str, Path],
        header: bool = True,
        quotechar: str = '"',
        delimiter: str = ',',
    ):
        """
        Load data into an Omniscidb table from CSV file.

        Wraps the COPY FROM DML statement.

        Parameters
        ----------
        path : str or pathlib.Path
          Path to the input data file
        header : bool, optional, default True
          Indicating whether the input file has a header line
        quotechar : str, optional, default '"'
          The character used to denote the start and end of a quoted item.
        delimiter : str, optional, default ','

        Returns
        -------
        query : OmniSciDBQuery
        """
        kwargs = {
            'header': header,
            'quote': quotechar,
            'quoted': True if quotechar != '' else False,
            'delimiter': delimiter,
        }
        stmt = ddl.LoadData(self._qualified_name, path, **kwargs)
        return self._execute(stmt)

    @property
    def name(self) -> str:
        """Return the operation name.

        Returns
        -------
        str
        """
        return self.op().name

    def rename(self, new_name, database=None):
        """
        Rename table to a given name.

        Parameters
        ----------
        new_name : string
        database : string

        Returns
        -------
        renamed : OmniSciDBTable
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
        # internal function that runs DDL operation
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


class OmniSciDBClient(SQLClient):
    """Client class for OmniSciDB backend."""

    database_class = Database
    query_class = OmniSciDBQuery
    dialect = OmniSciDBDialect
    table_expr_class = OmniSciDBTable

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        host: str = None,
        port: str = 6274,
        database: str = None,
        protocol: str = 'binary',
        session_id: str = None,
        execution_type: str = EXECUTION_TYPE_CURSOR,
    ):
        """Initialize OmniSciDB Client.

        Parameters
        ----------
        uri : str, optional
        user : str, optional
        password : str, optional
        host : str, optional
        port : int, default 6274
        database : str, optional
        protocol : {'binary', 'http', 'https'}, default binary
        session_id: str, optional
        execution_type : {
          EXECUTION_TYPE_ICP, EXECUTION_TYPE_ICP_GPU, EXECUTION_TYPE_CURSOR
        }, default EXECUTION_TYPE_CURSOR

        Raises
        ------
        Exception
            if the given execution_type is not valid.
        PyMapDVersionError
            if session_id is given but pymapd version is less or equal to 0.12
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = database
        self.protocol = protocol
        self.session_id = session_id

        if execution_type not in (
            EXECUTION_TYPE_ICP,
            EXECUTION_TYPE_ICP_GPU,
            EXECUTION_TYPE_CURSOR,
        ):
            raise Exception('Execution type defined not available.')

        self.execution_type = execution_type

        if session_id:
            if self.version < pkg_resources.parse_version('0.12.0'):
                raise PyMapDVersionError(
                    'Must have pymapd > 0.12 to use session ID'
                )
            self.con = pymapd.connect(
                uri=uri,
                host=host,
                port=port,
                protocol=protocol,
                sessionid=session_id,
            )
        else:
            self.con = pymapd.connect(
                uri=uri,
                user=user,
                password=password,
                host=host,
                port=port,
                dbname=database,
                protocol=protocol,
            )

    def __del__(self):
        """Close the connection when instance is deleted."""
        self.close()

    def __enter__(self, **kwargs):
        """Update internal attributes when using `with` statement."""
        self.__dict__.update(**kwargs)
        return self

    def __exit__(self, *args):
        """Close the connection when exits the `with` statement."""
        self.close()

    def log(self, msg: str):
        """Print or log a message.

        Parameters
        ----------
        msg : string
        """
        log(msg)

    def close(self):
        """Close OmniSciDB connection and drop any temporary objects."""
        self.con.close()

    def _adapt_types(self, descr):
        names = []
        adapted_types = []
        for col in descr:
            names.append(col.name)
            adapted_types.append(
                OmniSciDBDataType._omniscidb_to_ibis_dtypes[col.type]
            )
        return names, adapted_types

    def _build_ast(self, expr, context):
        result = build_ast(expr, context)
        return result

    def _fully_qualified_name(self, name, database):
        # OmniSciDB raises error sometimes with qualified names
        return name

    def _get_list(self, cur):
        tuples = cur.cursor.fetchall()
        return [v[0] for v in tuples]

    def _get_schema_using_query(self, query):
        with self._execute(query, results=True) as result:
            # resets the state of the cursor and closes operation
            result.cursor.fetchall()
            names, ibis_types = self._adapt_types(
                _extract_column_details(result.cursor._result.row_set.row_desc)
            )

        return sch.Schema(names, ibis_types)

    def _get_schema_using_validator(self, query):
        result = self.con._client.sql_validate(self.con._session, query)
        return sch.Schema.from_tuples(
            (
                r,
                OmniSciDBDataType._omniscidb_to_ibis_dtypes[
                    pymapd_dtype._VALUES_TO_NAMES[result[r].col_type.type]
                ],
            )
            for r in result
        )

    def _get_table_schema(self, table_name, database=None):
        """Get table schema.

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
        """Execute a query.

        Paramters
        ---------
        query : DDL or DML or string

        Returns
        -------
        result : pandas.DataFrame

        Raises
        ------
        Exception
            if execution method fails.
        """
        if isinstance(query, (DDL, DML)):
            query = query.compile()

        if self.execution_type == EXECUTION_TYPE_ICP:
            execute = self.con.select_ipc
        elif self.execution_type == EXECUTION_TYPE_ICP_GPU:
            execute = self.con.select_ipc_gpu
        else:
            execute = self.con.cursor().execute

        cursor = (
            OmniSciDBGeoCursor
            if FULL_GEO_SUPPORTED
            else OmniSciDBDefaultCursor
        )

        try:
            result = cursor(execute(query))
        except Exception as e:
            raise Exception('{}: {}'.format(e, query))

        if results:
            return result

    def create_database(self, name, owner=None):
        """
        Create a new OmniSciDB database.

        Parameters
        ----------
        name : string
          Database name
        """
        statement = ddl.CreateDatabase(name, owner=owner)
        self._execute(statement)

    def describe_formatted(self, name: str) -> pd.DataFrame:
        """Describe a given table name.

        Parameters
        ----------
        name : string

        Returns
        -------
        pandas.DataFrame
        """
        return pd.DataFrame(
            [
                (
                    col.name,
                    OmniSciDBDataType.parse(col.type),
                    col.precision,
                    col.scale,
                    col.comp_param,
                    col.encoding,
                )
                for col in self.con.get_table_details(name)
            ],
            columns=[
                'column_name',
                'type',
                'precision',
                'scale',
                'comp_param',
                'encoding',
            ],
        )

    def drop_database(self, name, force=False):
        """
        Drop an OmniSciDB database.

        Parameters
        ----------
        name : string
          Database name
        force : boolean, default False
          If False and there are any tables in this database, raises an
          IntegrityError

        Raises
        ------
        ibis.common.exceptions.IntegrityError
            if given database has tables and force is not define as True
        """
        tables = []

        if not force or self.database(name):
            tables = self.list_tables(database=name)

        if not force and len(tables):
            raise com.IntegrityError(
                'Database {0} must be empty before being dropped, or set '
                'force=True'.format(name)
            )
        statement = ddl.DropDatabase(name)
        self._execute(statement)

    def create_user(self, name, password, is_super=False):
        """
        Create a new OmniSciDB user.

        Parameters
        ----------
        name : string
          User name
        password : string
          Password
        is_super : bool
          if user is a superuser
        """
        statement = ddl.CreateUser(
            name=name, password=password, is_super=is_super
        )
        self._execute(statement)

    def alter_user(
        self, name, password=None, is_super=None, insert_access=None
    ):
        """
        Alter OmniSciDB user parameters.

        Parameters
        ----------
        name : string
          User name
        password : string
          Password
        is_super : bool
          If user is a superuser
        insert_access : string
          If users need to insert records to a database they do not own,
          use insert_access property to give them the required privileges.
        """
        statement = ddl.AlterUser(
            name=name,
            password=password,
            is_super=is_super,
            insert_access=insert_access,
        )
        self._execute(statement)

    def drop_user(self, name):
        """
        Drop a given user.

        Parameters
        ----------
        name : string
          User name
        """
        statement = ddl.DropUser(name)
        self._execute(statement)

    def create_view(self, name, expr, database=None):
        """
        Create a view with a given name from a table expression.

        Parameters
        ----------
        name : string
        expr : ibis TableExpr
        database : string, optional
        """
        ast = self._build_ast(expr, OmniSciDBDialect.make_context())
        select = ast.queries[0]
        statement = ddl.CreateView(name, select, database=database)
        self._execute(statement)

    def drop_view(self, name, database=None):
        """
        Drop a given view.

        Parameters
        ----------
        name : string
        database : string, default None
        """
        statement = ddl.DropView(name, database=database)
        self._execute(statement, False)

    def create_table(
        self, table_name, obj=None, schema=None, database=None, max_rows=None
    ):
        """
        Create a new table from an Ibis table expression.

        Parameters
        ----------
        table_name : string
        obj : TableExpr or pandas.DataFrame, optional
          If passed, creates table from select statement results
        schema : ibis.Schema, optional
          Mutually exclusive with expr, creates an empty table with a
          particular schema
        database : string, optional
        max_rows : int, optional
          Set the maximum number of rows allowed in a table to create a capped
          collection. When this limit is reached, the oldest fragment is
          removed. Default = 2^62.

        Examples
        --------
        >>> con.create_table('new_table_name', table_expr)  # doctest: +SKIP
        """
        _database = self.db_name
        self.set_database(database)

        if obj is not None:
            if isinstance(obj, pd.DataFrame):
                raise NotImplementedError(
                    'Pandas Data Frame input not implemented.'
                )
            else:
                to_insert = obj
            ast = self._build_ast(to_insert, OmniSciDBDialect.make_context())
            select = ast.queries[0]

            statement = ddl.CTAS(table_name, select, database=database)
        elif schema is not None:
            statement = ddl.CreateTableWithSchema(
                table_name, schema, database=database, max_rows=max_rows
            )
        else:
            raise com.IbisError('Must pass expr or schema')

        self._execute(statement, False)
        self.set_database(_database)

    def drop_table(self, table_name, database=None, force=False):
        """
        Drop a given table.

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
        _database = self.db_name
        self.set_database(database)

        statement = ddl.DropTable(
            table_name, database=database, must_exist=not force
        )
        self._execute(statement, False)
        self.set_database(_database)

    def truncate_table(self, table_name, database=None):
        """
        Delete all rows from, but do not drop, an existing table.

        Parameters
        ----------
        table_name : string
        database : string, optional
        """
        statement = ddl.TruncateTable(table_name, database=database)
        self._execute(statement, False)

    def drop_table_or_view(
        self, name: str, database: str = None, force: bool = False
    ):
        """Attempt to drop a relation that may be a view or table.

        Parameters
        ----------
        name : str
        database : str, optional
        force : bool, optional

        Raises
        ------
        Exception
            if the drop operation fails.
        """
        try:
            self.drop_table(name, database=database)
        except Exception as e:
            try:
                self.drop_view(name, database=database)
            except Exception:
                raise e

    def database(self, name=None):
        """Connect to a given database.

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
                uri=self.uri,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=name,
                protocol=self.protocol,
                session_id=self.session_id,
                execution_type=self.execution_type,
            )
            return self.database_class(name, new_client)

    def load_data(self, table_name, obj, database=None, **kwargs):
        """Load data into a given table.

        Wraps the LOAD DATA DDL statement. Loads data into an OmniSciDB table
        by physically moving data files.

        Parameters
        ----------
        table_name : string
        obj: pandas.DataFrame or pyarrow.Table
        database : string, optional
        """
        _database = self.db_name
        self.set_database(database)
        self.con.load_table(table_name, obj, **kwargs)
        self.set_database(_database)

    @property
    def current_database(self):
        """Get the current database name."""
        return self.db_name

    def set_database(self, name: str):
        """Set a given database for the current connect.

        Parameters
        ----------
        name : string
        """
        if self.db_name != name and name is not None:
            self.con.close()
            self.con = pymapd.connect(
                uri=self.uri,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                dbname=name,
                protocol=self.protocol,
                sessionid=self.session_id,
            )
            self.db_name = name

    @com.mark_as_unsupported
    def exists_database(self, name: str):
        """Check if the given database exists.

        Parameters
        ----------
        name : str

        Raises
        ------
        NotImplementedError
            Method not supported yet.
        """

    @com.mark_as_unsupported
    def list_databases(self, like: str = None):
        """List all databases.

        Parameters
        ----------
        like : str, optional

        Raises
        ------
        NotImplementedError
            Method not supported yet.
        """

    def exists_table(self, name: str, database: str = None):
        """
        Determine if the indicated table or view exists.

        Parameters
        ----------
        name : string
        database : string, default None

        Returns
        -------
        if_exists : boolean
        """
        return bool(self.list_tables(like=name, database=database))

    def list_tables(self, like: str = None, database: str = None) -> list:
        """List all tables inside given or current database.

        Parameters
        ----------
        like : str, optional
        database : str, optional

        Returns
        -------
        list
        """
        _database = None

        if not self.db_name == database:
            _database = self.db_name
            self.set_database(database)

        tables = self.con.get_tables()

        if _database:
            self.set_database(_database)

        if like is None:
            return tables
        pattern = re.compile(like)
        return list(filter(lambda t: pattern.findall(t), tables))

    def get_schema(self, table_name, database=None):
        """
        Return a Schema object for the given table and database.

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
            col_types.append(OmniSciDBDataType.parse(col.type))

        return sch.schema(
            [
                (col.name, OmniSciDBDataType.parse(col.type))
                for col in self.con.get_table_details(table_name)
            ]
        )

    def sql(self, query: str):
        """
        Convert a SQL query to an Ibis table expression.

        Parameters
        ----------
        query : string

        Returns
        -------
        table : TableExpr
        """
        # Remove `;` + `--` (comment)
        query = re.sub(r'\s*;\s*--', '\n--', query.strip())
        # Remove trailing ;
        query = re.sub(r'\s*;\s*$', '', query.strip())
        schema = self._get_schema_using_validator(query)
        return ops.SQLQueryResult(query, schema, self).to_expr()

    @property
    def version(self):
        """Return the backend library version.

        Returns
        -------
        string
            Version of the backend library.
        """
        # pymapd doesn't have __version__
        dist = pkg_resources.get_distribution('pymapd')
        return pkg_resources.parse_version(dist.version)


@dt.dtype.register(OmniSciDBDataType)
def omniscidb_to_ibis_dtype(omniscidb_dtype):
    """
    Register OmniSciDB Data Types.

    Parameters
    ----------
    omniscidb_dtype : OmniSciDBDataType

    Returns
    -------
    ibis.expr.datatypes.DataType
    """
    return omniscidb_dtype.to_ibis()
