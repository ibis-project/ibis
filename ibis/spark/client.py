import pandas as pd
import pyspark as ps
import pyspark.sql.types as pt
import regex as re
from pkg_resources import parse_version

import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.client import Database, Query, SQLClient
from ibis.spark import compiler as comp
from ibis.spark import ddl
from ibis.spark.compiler import SparkDialect, build_ast
from ibis.util import log

# maps pyspark type class to ibis type class
_SPARK_DTYPE_TO_IBIS_DTYPE = {
    pt.NullType : dt.Null,
    pt.StringType : dt.String,
    pt.BinaryType : dt.Binary,
    pt.BooleanType : dt.Boolean,
    pt.DateType : dt.Date,
    pt.DoubleType : dt.Double,
    pt.FloatType : dt.Float,
    pt.ByteType : dt.Int8,
    pt.IntegerType : dt.Int32,
    pt.LongType : dt.Int64,
    pt.ShortType : dt.Int16,
}


@dt.dtype.register(pt.DataType)
def spark_dtype_to_ibis_dtype(spark_type_obj, nullable=True):
    """Convert Spark SQL type objects to ibis type objects."""
    ibis_type_class = _SPARK_DTYPE_TO_IBIS_DTYPE.get(type(spark_type_obj))
    return ibis_type_class(nullable=nullable)


@dt.dtype.register(pt.TimestampType)
def spark_timestamp_dtype_to_ibis_dtype(spark_type_obj, nullable=True):
    return dt.Timestamp(nullable=nullable)


@dt.dtype.register(pt.DecimalType)
def spark_decimal_dtype_to_ibis_dtype(spark_type_obj, nullable=True):
    precision = spark_type_obj.precision
    scale = spark_type_obj.scale
    return dt.Decimal(precision, scale, nullable=nullable)


@dt.dtype.register(pt.ArrayType)
def spark_array_dtype_to_ibis_dtype(spark_type_obj, nullable=True):
    value_type = dt.dtype(
        spark_type_obj.elementType,
        nullable=spark_type_obj.containsNull
    )
    return dt.Array(value_type, nullable=nullable)


@dt.dtype.register(pt.MapType)
def spark_map_dtype_to_ibis_dtype(spark_type_obj, nullable=True):
    key_type = dt.dtype(spark_type_obj.keyType)
    value_type = dt.dtype(
        spark_type_obj.valueType,
        nullable=spark_type_obj.valueContainsNull
    )
    return dt.Map(key_type, value_type, nullable=nullable)


@dt.dtype.register(pt.StructType)
def spark_struct_dtype_to_ibis_dtype(spark_type_obj, nullable=True):
    names = spark_type_obj.names
    fields = spark_type_obj.fields
    ibis_types = [dt.dtype(f.dataType, nullable=f.nullable) for f in fields]
    return dt.Struct(names, ibis_types, nullable=nullable)


@sch.infer.register(ps.sql.dataframe.DataFrame)
def spark_dataframe_schema(df):
    """Infer the schema of a Spark SQL `DataFrame` object."""
    # df.schema is a pt.StructType
    schema_struct = dt.dtype(df.schema)

    return sch.schema(schema_struct.names, schema_struct.types)


class SparkCursor:
    """Spark cursor.

    This allows the Spark client to reuse machinery in
    :file:`ibis/client.py`.

    """

    def __init__(self, query):
        """

        Construct a SparkCursor with query `query`.

        Parameters
        ----------
        query : pyspark.sql.DataFrame
          Contains result of query.

        """
        self.query = query

    def fetchall(self):
        """Fetch all rows."""
        result = self.query.collect()  # blocks until finished
        return result

    @property
    def columns(self):
        """Return the columns of the result set."""
        return self.query.columns

    @property
    def description(self):
        """Get the fields of the result set's schema."""
        return self.query.schema

    def __enter__(self):
        # For compatibility when constructed from Query.execute()
        """No-op for compatibility.

        See Also
        --------
        ibis.client.Query.execute

        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """No-op for compatibility.

        See Also
        --------
        ibis.client.Query.execute

        """


class SparkQuery(Query):

    def _fetch(self, cursor):
        df = cursor.query.toPandas()  # blocks until finished
        schema = self.schema()
        return schema.apply_to(df)


class SparkDatabase(Database):
    pass


class SparkDatabaseTable(ops.DatabaseTable):
    pass


class SparkTable(ir.TableExpr):

    @property
    def _qualified_name(self):
        return self.op().args[0]

    def _match_name(self):
        m = ddl.fully_qualified_re.match(self._qualified_name)
        if not m:
            return None, self._qualified_name
        db, quoted, unquoted = m.groups()
        return db, quoted or unquoted

    @property
    def _database(self):
        return self._match_name()[0]

    @property
    def _unqualified_name(self):
        return self._match_name()[1]

    @property
    def name(self):
        return self.op().name

    @property
    def _client(self):
        return self.op().source

    def _execute(self, stmt):
        return self._client._execute(stmt)

    def compute_stats(self, noscan=False):
        """
        Invoke Spark ANALYZE TABLE <tbl> COMPUTE STATISTICS command to
        compute column, table, and partition statistics.

        See also SparkClient.compute_stats
        """
        return self._client.compute_stats(
            self._qualified_name, noscan=noscan
        )

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
        values=None,
        validate=True,
    ):
        """
        Insert into Spark table.

        Parameters
        ----------
        obj : TableExpr or pandas DataFrame
        overwrite : boolean, default False
          If True, will replace existing contents of table
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
            spark_df = self._session.createDataFrame(obj)
            spark_df.insertInto(self.name, overwrite=overwrite)
            return

        expr = obj

        if values is not None:
            raise NotImplementedError

        if validate:
            existing_schema = self.schema()
            insert_schema = expr.schema()
            if not insert_schema.equals(existing_schema):
                _validate_compatible(insert_schema, existing_schema)

        ast = build_ast(expr, SparkDialect.make_context())
        select = ast.queries[0]
        statement = ddl.InsertSelect(
            self._qualified_name,
            select,
            overwrite=overwrite,
        )
        return self._execute(statement.compile())

    def rename(self, new_name):
        """
        Rename table inside Spark. References to the old table are no longer
        valid. Spark does not support moving tables across databases using
        rename.

        Parameters
        ----------
        new_name : string

        Returns
        -------
        renamed : SparkTable
        """
        new_qualified_name = _fully_qualified_name(new_name, self._database)

        statement = ddl.RenameTable(
            self._qualified_name, new_name
        )
        self._client._execute(statement.compile())

        op = self.op().change_name(new_qualified_name)
        return type(self)(op)

    def alter(
        self,
        tbl_properties=None,
    ):
        """
        Change setting and parameters of the table.

        Parameters
        ----------
        tbl_properties : dict, optional

        Returns
        -------
        None (for now)
        """

        stmt = ddl.AlterTable(
            self._qualified_name,
            tbl_properties=tbl_properties
        )
        return self._execute(stmt.compile())


class SparkClient(SQLClient):

    """
    An Ibis client interface that uses Spark SQL.
    """

    dialect = comp.SparkDialect
    database_class = SparkDatabase
    query_class = SparkQuery
    table_class = SparkDatabaseTable
    table_expr_class = SparkTable

    def __init__(self, **kwargs):
        self._context = ps.SparkContext(**kwargs)
        self._session = ps.sql.SparkSession(self._context)
        self._catalog = self._session.catalog

    def close(self):
        """
        Close Spark connection and drop any temporary objects
        """
        self._context.stop()

    def _build_ast(self, expr, context):
        result = comp.build_ast(expr, context)
        return result

    def _execute(self, stmt, results=False):
        query = self._session.sql(stmt)
        if results:
            return SparkCursor(query)

    def database(self, name=None):
        return self.database_class(name or self.current_database, self)

    @property
    def current_database(self):
        """
        String name of the current database.
        """
        return self._catalog.currentDatabase()

    def _get_table_schema(self, table_name):
        return self.get_schema(table_name)

    def _get_schema_using_query(self, query):
        cur = self._execute(query, results=True)
        return spark_dataframe_schema(cur.query)

    def log(self, msg):
        log(msg)

    def table(self, name, database=None):
        """
        Create a table expression that references a particular table or view
        in the database.

        Parameters
        ----------
        name : string
        database : string, optional

        Returns
        -------
        table : TableExpr
        """
        qualified_name = _fully_qualified_name(name, database)
        if not database:
            try:
                self._session.table(qualified_name)
            except ps.sql.utils.AnalysisException:
                qualified_name = _fully_qualified_name(
                    name, self.current_database
                )
                try:
                    self._session.table(qualified_name)
                except ps.sql.utils.AnalysisException as e:
                    raise e
        schema = self._get_table_schema(qualified_name)
        node = self.table_class(qualified_name, schema, self)
        return self.table_expr_class(node)

    def list_functions(self, database=None):
        return self._catalog.listFunctions(dbName=None)

    def list_tables(self, like=None, database=None):
        """
        List tables in the current (or indicated) database. Like the SHOW
        TABLES command.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'
        database : string, default None
          If not passed, uses the current/default database

        Returns
        -------
        results : list of strings
        """
        results = [t.name for t in self._catalog.listTables(dbName=database)]
        if like:
            results = [
                table_name
                for table_name in results
                if re.match(like, table_name) is not None
            ]

        return results

    def set_database(self, name):
        """
        Set the default database scope for client
        """
        self._catalog.setCurrentDatabase(name)

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

    def list_databases(self, like=None):
        """
        List databases in the Spark SQL cluster.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'

        Returns
        -------
        results : list of strings
        """
        results = [db.name for db in self._catalog.listDatabases()]
        if like:
            results = [
                database_name
                for database_name in results
                if re.match(like, database_name) is not None
            ]

        return results

    def create_database(self, name, path=None, force=False):
        """
        Create a new Spark database

        Parameters
        ----------
        name : string
          Database name
        path : string, default None
          Path where to store the database data; otherwise uses Spark
          default
        """
        statement = ddl.CreateDatabase(name, path=path, can_exist=force)
        return self._execute(statement.compile())

    def drop_database(self, name, force=False):
        """Drop a Spark database.

        Parameters
        ----------
        name : string
          Database name
        force : bool, default False
          If False and there are any tables in this database, raises an
          IntegrityError and Spark throws exception if database does not exist
        """
        if not force:
            tables = self.list_tables(database=name)
            functions = self.list_functions(database=name)
            if len(tables) > 0 or len(functions) > 0:
                raise com.IntegrityError(
                    'Database {0} must be empty before '
                    'being dropped, or set '
                    'force=True'.format(name)
                )
        statement = ddl.DropDatabase(name, must_exist=not force, cascade=force)
        return self._execute(statement.compile())

    def get_schema(self, table_name, database=None):
        """
        Return a Schema object for the indicated table and database

        Parameters
        ----------
        table_name : string
          May be fully qualified
        database : string
          Spark does not have a database argument for its table() method,
          so this must be None

        Returns
        -------
        schema : ibis Schema
        """
        if database is not None:
            raise com.UnsupportedArgumentError(
                'Spark does not support database param for table'
            )

        df = self._session.table(table_name)

        return sch.infer(df)

    @property
    def version(self):
        return parse_version(ps.__version__)

    def create_table(
        self,
        table_name,
        obj=None,
        schema=None,
        database=None,
        force=False,
        # HDFS options
        format='parquet',
        location=None,
    ):
        """
        Create a new table in Spark using an Ibis table expression.

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
          If true, create table if table with indicated name already exists
        format : {'parquet'}
        location : string, default None
          Specify the directory location where Spark reads and writes files
          for the table
        Examples
        --------
        >>> con.create_table('new_table_name', table_expr)  # doctest: +SKIP
        """
        if obj is not None:
            if isinstance(obj, pd.DataFrame):
                spark_df = self._session.createDataFrame(obj)
                mode = 'error'
                if force:
                    mode = 'overwrite'
                spark_df.write.saveAsTable(
                    table_name,
                    format=format,
                    mode=mode,
                )
                return self.table(table_name, database=database)

            ast = self._build_ast(obj, SparkDialect.make_context())
            select = ast.queries[0]

            statement = ddl.CTAS(
                table_name,
                select,
                database=database,
                can_exist=force,
                format=format,
                path=location,
            )
        elif schema is not None:
            statement = ddl.CreateTableWithSchema(
                table_name,
                schema,
                database=database,
                format=format,
                can_exist=force,
                path=location,
            )
        else:
            raise com.IbisError('Must pass expr or schema')

        return self._execute(statement.compile())

    def create_view(
        self,
        name,
        expr,
        database=None,
        or_replace=True,
        temporary=True,
    ):
        """
        Create a Spark view from a table expression

        Parameters
        ----------
        name : string
        expr : ibis TableExpr
        database : string, default None
        or_replace : boolean, default True
          Replace an existing view of the same name if it exists
        """
        ast = self._build_ast(expr, SparkDialect.make_context())
        select = ast.queries[0]
        statement = ddl.CreateView(
            name,
            select,
            database=database,
            or_replace=or_replace,
            temporary=temporary,
        )
        return self._execute(statement.compile())

    def drop_table(self, name, database=None, force=False):
        self.drop_table_or_view(name, database, force)

    def drop_view(self, name, database=None, force=False):
        self.drop_table_or_view(name, database, force)

    def drop_table_or_view(self, name, database=None, force=False):
        """
        Drop a Spark table or view

        Parameters
        ----------
        name : string
        database : string, default None (optional)
        force : boolean, default False
          Database may throw exception if table does not exist

        Examples
        --------
        >>> table = 'my_table'
        >>> db = 'operations'
        >>> con.drop_table_or_view(table, db, force=True)  # doctest: +SKIP
        """
        statement = ddl.DropTable(
            name, database=database, must_exist=not force
        )
        self._execute(statement.compile())

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

    def truncate_table(self, table_name, database=None):
        """
        Delete all rows from, but do not drop, an existing table

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        """
        statement = ddl.TruncateTable(table_name, database=database)
        self._execute(statement.compile())

    def insert(
        self,
        table_name,
        obj=None,
        database=None,
        overwrite=False,
        values=None,
        validate=True,
    ):
        """
        Insert into existing table.

        See SparkTable.insert for other parameters.

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
            values=values,
            validate=validate,
        )

    def compute_stats(self, name, database=None, noscan=False):
        """
        Issue COMPUTE STATISTICS command for a given table

        Parameters
        ----------
        name : string
          Can be fully qualified (with database name)
        database : string, optional
        noscan : boolean, default False
          If True, collect only basic statistics for the table (number of
          rows, size in bytes).
        """
        maybe_noscan = ' NOSCAN' if noscan else ''
        stmt = 'ANALYZE TABLE {0} COMPUTE STATISTICS{1}'.format(
            _fully_qualified_name(name, database),
            maybe_noscan
        )
        return self._execute(stmt)


def _fully_qualified_name(name, database):
    if ddl._is_fully_qualified(name):
        return name
    if database:
        return '{0}.`{1}`'.format(database, name)
    return name


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
