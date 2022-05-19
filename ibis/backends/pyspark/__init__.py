from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd
import pyspark
from pyspark.sql import DataFrame
from pyspark.sql.column import Column

if TYPE_CHECKING:
    import ibis.expr.types as ir
    import ibis.expr.operations as ops

import ibis.common.exceptions as com
import ibis.expr.schema as sch
import ibis.expr.types as types
import ibis.util as util
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.base.sql.ddl import (
    CreateDatabase,
    DropTable,
    TruncateTable,
    is_fully_qualified,
)
from ibis.backends.pyspark import ddl
from ibis.backends.pyspark.client import PySparkTable, spark_dataframe_schema
from ibis.backends.pyspark.compiler import (
    PySparkDatabaseTable,
    PySparkExprTranslator,
)
from ibis.backends.pyspark.datatypes import spark_dtype
from ibis.expr.scope import Scope
from ibis.expr.timecontext import canonicalize_context, localize_context

_read_csv_defaults = {
    'header': True,
    'multiLine': True,
    'mode': 'FAILFAST',
    'escape': '"',
}


class _PySparkCursor:
    """Spark cursor.

    This allows the Spark client to reuse machinery in
    `ibis/backends/base/sql/client.py`.

    """

    def __init__(self, query: DataFrame) -> None:
        """Construct a cursor with query `query`.

        Parameters
        ----------
        query
            PySpark query
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
        """No-op for compatibility."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """No-op for compatibility."""


class Backend(BaseSQLBackend):
    name = 'pyspark'
    table_class = PySparkDatabaseTable
    table_expr_class = PySparkTable

    def do_connect(self, session: pyspark.sql.SparkSession) -> None:
        """Create a PySpark `Backend` for use with Ibis.

        Parameters
        ----------
        session
            A SparkSession instance

        Examples
        --------
        >>> import ibis
        >>> import pyspark
        >>> session = pyspark.sql.SparkSession.builder.getOrCreate()
        >>> ibis.pyspark.connect(session)
        """
        self._context = session.sparkContext
        self._session = session
        self._catalog = session.catalog

        # Spark internally stores timestamps as UTC values, and timestamp data
        # that is brought in without a specified time zone is converted as
        # local time to UTC with microsecond resolution.
        # https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#timestamp-with-time-zone-semantics
        self._session.conf.set('spark.sql.session.timeZone', 'UTC')

    @property
    def version(self):
        return pyspark.__version__

    @util.deprecated(
        version='2.0',
        instead='use a new connection to the database',
    )
    def set_database(self, name):
        self._catalog.setCurrentDatabase(name)

    @property
    def current_database(self):
        return self._catalog.currentDatabase()

    def list_databases(self, like=None):
        databases = [db.name for db in self._catalog.listDatabases()]
        return self._filter_with_like(databases, like)

    def list_tables(self, like=None, database=None):
        tables = [
            t.name
            for t in self._catalog.listTables(
                dbName=database or self.current_database
            )
        ]
        return self._filter_with_like(tables, like)

    #
    def compile(self, expr, timecontext=None, params=None, *args, **kwargs):
        """Compile an ibis expression to a PySpark DataFrame object."""

        if timecontext is not None:
            session_timezone = self._session.conf.get(
                'spark.sql.session.timeZone'
            )
            # Since spark use session timezone for tz-naive timestamps
            # we localize tz-naive context here to match that behavior
            timecontext = localize_context(
                canonicalize_context(timecontext), session_timezone
            )

        # Insert params in scope
        if params is None:
            scope = Scope()
        else:
            scope = Scope(
                {param.op(): raw_value for param, raw_value in params.items()},
                timecontext,
            )
        return PySparkExprTranslator().translate(
            expr, scope=scope, timecontext=timecontext
        )

    def execute(
        self,
        expr: ir.Expr,
        timecontext: Mapping | None = None,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: str = 'default',
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""
        if isinstance(expr, types.Table):
            return self.compile(expr, timecontext, params, **kwargs).toPandas()
        elif isinstance(expr, types.Column):
            # expression must be named for the projection
            if not expr.has_name():
                expr = expr.name("tmp")
            return self.compile(
                expr.to_projection(), timecontext, params, **kwargs
            ).toPandas()[expr.get_name()]
        elif isinstance(expr, types.Scalar):
            compiled = self.compile(expr, timecontext, params, **kwargs)
            if isinstance(compiled, Column):
                # attach result column to a fake DataFrame and
                # select the result
                compiled = self._session.range(0, 1).select(compiled)
            return compiled.toPandas().iloc[0, 0]
        else:
            raise com.IbisError(
                f"Cannot execute expression of type: {type(expr)}"
            )

    @staticmethod
    def _fully_qualified_name(name, database):
        if is_fully_qualified(name):
            return name
        if database:
            return f'{database}.`{name}`'
        return name

    def close(self):
        """Close Spark connection and drop any temporary objects."""
        self._context.stop()

    def fetch_from_cursor(self, cursor, schema):
        df = cursor.query.toPandas()  # blocks until finished
        return schema.apply_to(df)

    def raw_sql(self, query: str) -> _PySparkCursor:
        query = self._session.sql(query)
        return _PySparkCursor(query)

    def _get_schema_using_query(self, query):
        cur = self.raw_sql(query)
        return spark_dataframe_schema(cur.query)

    def _get_jtable(self, name, database=None):
        try:
            jtable = self._catalog._jcatalog.getTable(
                self._fully_qualified_name(name, database)
            )
        except pyspark.sql.utils.AnalysisException as e:
            raise com.IbisInputError(str(e)) from e
        return jtable

    def table(self, name: str, database: str | None = None) -> ir.Table:
        """Return a table expression from a table or view in the database.

        Parameters
        ----------
        name
            Table name
        database
            Database in which the table resides

        Returns
        -------
        Table
            Table named `name` from `database`
        """
        jtable = self._get_jtable(name, database)
        name, database = jtable.name(), jtable.database()

        qualified_name = self._fully_qualified_name(name, database)

        schema = self.get_schema(qualified_name)
        node = self.table_class(qualified_name, schema, self)
        return self.table_expr_class(node)

    def create_database(
        self,
        name: str,
        path: str | Path | None = None,
        force: bool = False,
    ) -> Any:
        """Create a new Spark database.

        Parameters
        ----------
        name
            Database name
        path
            Path where to store the database data; otherwise uses Spark default
        """
        statement = CreateDatabase(name, path=path, can_exist=force)
        return self.raw_sql(statement.compile())

    def drop_database(self, name: str, force: bool = False) -> Any:
        """Drop a Spark database.

        Parameters
        ----------
        name
            Database name
        force
            If False, Spark throws exception if database is not empty or
            database does not exist
        """
        statement = ddl.DropDatabase(name, must_exist=not force, cascade=force)
        return self.raw_sql(statement.compile())

    def get_schema(
        self,
        table_name: str,
        database: str | None = None,
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            Table name. May be fully qualified
        database
            Spark does not have a database argument for its table() method,
            so this must be None

        Returns
        -------
        Schema
            An ibis schema
        """
        if database is not None:
            raise com.UnsupportedArgumentError(
                'Spark does not support the `database` argument for '
                '`get_schema`'
            )

        df = self._session.table(table_name)

        return sch.infer(df)

    def _schema_from_csv(self, path, **kwargs):
        """
        Return a Schema object for the indicated csv file. Spark goes through
        the file once to determine the schema. See documentation for
        `pyspark.sql.DataFrameReader` for kwargs.

        Parameters
        ----------
        path
            Path to CSV

        Returns
        -------
        schema : ibis Schema
        """
        options = _read_csv_defaults.copy()
        options.update(kwargs)
        options['inferSchema'] = True

        df = self._session.read.csv(path, **options)
        return spark_dataframe_schema(df)

    def _create_table_or_temp_view_from_csv(
        self,
        name: str,
        path: Path,
        schema: sch.Schema | None = None,
        database: str | None = None,
        force: bool = False,
        temp_view: bool = False,
        format: str = 'parquet',
        **kwargs: Any,
    ):
        options = _read_csv_defaults.copy()
        options.update(kwargs)

        if schema:
            assert ('inferSchema', True) not in options.items()
            schema = spark_dtype(schema)
            options['schema'] = schema
        else:
            options['inferSchema'] = True

        df = self._session.read.csv(path, **options)

        if temp_view:
            if force:
                df.createOrReplaceTempView(name)
            else:
                df.createTempView(name)
        else:
            qualified_name = self._fully_qualified_name(
                name, database or self.current_database
            )
            mode = 'error'
            if force:
                mode = 'overwrite'
            df.write.saveAsTable(qualified_name, format=format, mode=mode)

    def create_table(
        self,
        table_name: str,
        obj: ir.Table | pd.DataFrame | None = None,
        schema: sch.Schema | None = None,
        database: str | None = None,
        force: bool = False,
        # HDFS options
        format: str = 'parquet',
    ):
        """Create a new table in Spark.

        Parameters
        ----------
        table_name
            Table name
        obj
            If passed, creates table from select statement results
        schema
            Mutually exclusive with obj, creates an empty table with a
            schema
        database
            Database name
        force
            If true, create table if table with indicated name already exists
        format
            Table format

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
                    table_name, format=format, mode=mode
                )
                return

            ast = self.compiler.to_ast(obj)
            select = ast.queries[0]

            statement = ddl.CTAS(
                table_name,
                select,
                database=database,
                can_exist=force,
                format=format,
            )
        elif schema is not None:
            statement = ddl.CreateTableWithSchema(
                table_name,
                schema,
                database=database,
                format=format,
                can_exist=force,
            )
        else:
            raise com.IbisError('Must pass expr or schema')

        return self.raw_sql(statement.compile())

    def create_view(
        self,
        name: str,
        expr: ir.Table,
        database: str | None = None,
        can_exist: bool = False,
        temporary: bool = False,
    ):
        """Create a Spark view from a table expression.

        Parameters
        ----------
        name
            View name
        expr
            Expression to use for the view
        database
            Database name
        can_exist
            Replace an existing view of the same name if it exists
        temporary
            Whether the table is temporary
        """
        ast = self.compiler.to_ast(expr)
        select = ast.queries[0]
        statement = ddl.CreateView(
            name,
            select,
            database=database,
            can_exist=can_exist,
            temporary=temporary,
        )
        return self.raw_sql(statement.compile())

    def drop_table(
        self,
        name: str,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table."""
        self.drop_table_or_view(name, database, force)

    def drop_view(
        self,
        name: str,
        database: str | None = None,
        force: bool = False,
    ):
        """Drop a view."""
        self.drop_table_or_view(name, database, force)

    def drop_table_or_view(
        self,
        name: str,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a Spark table or view.

        Parameters
        ----------
        name
            Table or view name
        database
            Database name
        force
            Database may throw exception if table does not exist

        Examples
        --------
        >>> table = 'my_table'
        >>> db = 'operations'
        >>> con.drop_table_or_view(table, db, force=True)  # doctest: +SKIP
        """
        statement = DropTable(name, database=database, must_exist=not force)
        self.raw_sql(statement.compile())

    def truncate_table(
        self,
        table_name: str,
        database: str | None = None,
    ) -> None:
        """Delete all rows from an existing table.

        Parameters
        ----------
        table_name
            Table name
        database
            Database name
        """
        statement = TruncateTable(table_name, database=database)
        self.raw_sql(statement.compile())

    def insert(
        self,
        table_name: str,
        obj: ir.Table | pd.DataFrame | None = None,
        database: str | None = None,
        overwrite: bool = False,
        values: Any | None = None,
        validate: bool = True,
    ) -> Any:
        """Insert data into an existing table.

        Examples
        --------
        >>> table = 'my_table'
        >>> con.insert(table, table_expr)  # doctest: +SKIP

        # Completely overwrite contents
        >>> con.insert(table, table_expr, overwrite=True)  # doctest: +SKIP
        """
        table = self.table(table_name, database=database)
        return table.insert(
            obj=obj, overwrite=overwrite, values=values, validate=validate
        )

    def compute_stats(
        self,
        name: str,
        database: str | None = None,
        noscan: bool = False,
    ) -> Any:
        """Issue a `COMPUTE STATISTICS` command for a given table.

        Parameters
        ----------
        name
            Table name
        database
            Database name
        noscan
            If True, collect only basic statistics for the table (number of
            rows, size in bytes).
        """
        maybe_noscan = ' NOSCAN' if noscan else ''
        stmt = 'ANALYZE TABLE {} COMPUTE STATISTICS{}'.format(
            self._fully_qualified_name(name, database), maybe_noscan
        )
        return self.raw_sql(stmt)

    def has_operation(cls, operation: type[ops.Value]) -> bool:
        return operation in PySparkExprTranslator._registry.keys()
