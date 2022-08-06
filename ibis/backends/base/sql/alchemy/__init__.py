from __future__ import annotations

import contextlib
import getpass
from typing import Any, Literal

import pandas as pd
import sqlalchemy as sa

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.base.sql.alchemy.database import (
    AlchemyDatabase,
    AlchemyTable,
)
from ibis.backends.base.sql.alchemy.datatypes import (
    schema_from_table,
    table_from_schema,
    to_sqla_type,
)
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.backends.base.sql.alchemy.query_builder import AlchemyCompiler
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    get_sqla_table,
    infix_op,
    reduction,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
    varargs,
    variance_reduction,
)
from ibis.backends.base.sql.alchemy.translator import (
    AlchemyContext,
    AlchemyExprTranslator,
)

__all__ = (
    'BaseAlchemyBackend',
    'AlchemyExprTranslator',
    'AlchemyContext',
    'AlchemyCompiler',
    'AlchemyTable',
    'AlchemyDatabase',
    'AlchemyContext',
    'sqlalchemy_operation_registry',
    'sqlalchemy_window_functions_registry',
    'reduction',
    'variance_reduction',
    'fixed_arity',
    'unary',
    'infix_op',
    'get_sqla_table',
    'to_sqla_type',
    'schema_from_table',
    'table_from_schema',
    'varargs',
)


class BaseAlchemyBackend(BaseSQLBackend):
    """Backend class for backends that compile to SQLAlchemy expressions."""

    database_class = AlchemyDatabase
    table_class = AlchemyTable
    compiler = AlchemyCompiler

    def _build_alchemy_url(
        self, url, host, port, user, password, database, driver
    ):
        if url is not None:
            return sa.engine.url.make_url(url)

        user = user or getpass.getuser()
        return sa.engine.url.URL.create(
            driver,
            host=host,
            port=port,
            username=user,
            password=password,
            database=database,
        )

    @property
    def _current_schema(self) -> str | None:
        return None

    def do_connect(self, con: sa.engine.Engine) -> None:
        self.con = con
        self._inspector = sa.inspect(self.con)
        self.meta = sa.MetaData(bind=self.con)
        self._schemas: dict[str, sch.Schema] = {}
        self._temp_views: set[str] = set()

    @property
    def version(self):
        return '.'.join(map(str, self.con.dialect.server_version_info))

    def list_tables(self, like=None, database=None):
        tables = self.inspector.get_table_names(schema=database)
        views = self.inspector.get_view_names(schema=database)
        return self._filter_with_like(tables + views, like)

    def list_databases(self, like=None):
        """List databases in the current server."""
        databases = self.inspector.get_schema_names()
        return self._filter_with_like(databases, like)

    @property
    def inspector(self):
        self._inspector.info_cache.clear()
        return self._inspector

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with self.begin() as con:
            yield con.execute(*args, **kwargs)

    @staticmethod
    def _to_geodataframe(df, schema):
        """Convert `df` to a `GeoDataFrame`.

        Required libraries for geospatial support must be installed and a
        geospatial column is present in the dataframe.
        """
        import geopandas as gpd
        from geoalchemy2 import shape

        geom_col = None
        for name, dtype in schema.items():
            if isinstance(dtype, dt.GeoSpatial):
                geom_col = geom_col or name
                df[name] = df[name].map(
                    lambda row: None if row is None else shape.to_shape(row)
                )
        if geom_col:
            df[geom_col] = gpd.array.GeometryArray(df[geom_col].values)
            df = gpd.GeoDataFrame(df, geometry=geom_col)
        return df

    def fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        df = pd.DataFrame.from_records(
            cursor,
            columns=cursor.keys(),
            coerce_float=True,
        )
        df = schema.apply_to(df)
        if len(df) and geospatial_supported:
            return self._to_geodataframe(df, schema)
        return df

    @contextlib.contextmanager
    def begin(self):
        with self.con.begin() as bind:
            yield bind

    def create_table(
        self,
        name: str,
        expr: pd.DataFrame | ir.Table | None = None,
        schema: sch.Schema | None = None,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Create a table.

        Parameters
        ----------
        name
            Table name to create
        expr
            DataFrame or table expression to use as the data source
        schema
            An ibis schema
        database
            A database
        force
            Check whether a table exists before creating it
        """
        if database == self.current_database:
            # avoid fully qualified name
            database = None

        if database is not None:
            raise NotImplementedError(
                'Creating tables from a different database is not yet '
                'implemented'
            )

        if expr is None and schema is None:
            raise ValueError('You must pass either an expression or a schema')

        if expr is not None and schema is not None:
            if not expr.schema().equals(ibis.schema(schema)):
                raise TypeError(
                    'Expression schema is not equal to passed schema. '
                    'Try passing the expression without the schema'
                )
        if schema is None:
            schema = expr.schema()

        self._schemas[self._fully_qualified_name(name, database)] = schema
        t = self._table_from_schema(
            name, schema, database=database or self.current_database
        )

        with self.begin() as bind:
            t.create(bind=bind, checkfirst=force)
            if expr is not None:
                compiled = self.compile(expr)
                insert = t.insert()
                if isinstance(expr.op(), ops.InMemoryTable):
                    compiled = compiled.get_final_froms()[0]
                    sa_expr = insert.values(*compiled._data)
                else:
                    sa_expr = insert.from_select(list(expr.columns), compiled)
                bind.execute(sa_expr)

    def _columns_from_schema(
        self, name: str, schema: sch.Schema
    ) -> list[sa.Column]:
        return [
            sa.Column(colname, to_sqla_type(dtype), nullable=dtype.nullable)
            for colname, dtype in zip(schema.names, schema.types)
        ]

    def _table_from_schema(
        self, name: str, schema: sch.Schema, database: str | None = None
    ) -> sa.Table:
        columns = self._columns_from_schema(name, schema)
        return sa.Table(name, self.meta, *columns)

    def drop_table(
        self,
        table_name: str,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table.

        Parameters
        ----------
        table_name
            Table to drop
        database
            Database to drop table from
        force
            Check for existence before dropping
        """
        if database == self.current_database:
            # avoid fully qualified name
            database = None

        if database is not None:
            raise NotImplementedError(
                'Dropping tables from a different database is not yet '
                'implemented'
            )

        t = self._get_sqla_table(table_name, schema=database, autoload=False)
        t.drop(checkfirst=force)

        assert not self.inspector.has_table(
            table_name
        ), f"Something went wrong during DROP of table {table_name!r}"

        self.meta.remove(t)

        qualified_name = self._fully_qualified_name(table_name, database)

        try:
            del self._schemas[qualified_name]
        except KeyError:  # schemas won't be cached if created with raw_sql
            pass

    def load_data(
        self,
        table_name: str,
        data: pd.DataFrame,
        database: str | None = None,
        if_exists: Literal['fail', 'replace', 'append'] = 'fail',
    ) -> None:
        """Load data from a dataframe to the backend.

        Parameters
        ----------
        table_name
            Name of the table in which to load data
        data
            Pandas DataFrame
        database
            Database in which the table exists
        if_exists
            What to do when data in `name` already exists

        Raises
        ------
        NotImplementedError
            Loading data to a table from a different database is not
            yet implemented
        """
        if database == self.current_database:
            # avoid fully qualified name
            database = None

        if database is not None:
            raise NotImplementedError(
                'Loading data to a table from a different database is not '
                'yet implemented'
            )

        data.to_sql(
            table_name,
            con=self.con,
            index=False,
            if_exists=if_exists,
            schema=self._current_schema,
        )

    def truncate_table(
        self,
        table_name: str,
        database: str | None = None,
    ) -> None:
        t = self._get_sqla_table(table_name, schema=database)
        t.delete().execute()

    def schema(self, name: str) -> sch.Schema:
        """Get an ibis schema from the current database for the table `name`.

        Parameters
        ----------
        name
            Table name

        Returns
        -------
        Schema
            The ibis schema of `name`
        """
        return self.database().schema(name)

    @property
    def current_database(self) -> str:
        """The name of the current database this client is connected to."""
        return self.database_name

    @util.deprecated(version='2.0', instead='use `list_databases`')
    def list_schemas(self, like: str | None = None) -> list[str]:
        return self.list_databases()

    def _log(self, sql):
        try:
            query_str = str(sql)
        except sa.exc.UnsupportedCompilationError:
            pass
        else:
            util.log(query_str)

    def _get_sqla_table(
        self,
        name: str,
        schema: str | None = None,
        autoload: bool = True,
        **kwargs: Any,
    ) -> sa.Table:
        return sa.Table(name, self.meta, schema=schema, autoload=autoload)

    def _sqla_table_to_expr(self, table: sa.Table) -> ir.Table:
        schema = self._schemas.get(table.name)
        node = self.table_class(
            source=self,
            sqla_table=table,
            name=table.name,
            schema=schema,
        )
        return self.table_expr_class(node)

    def table(
        self,
        name: str,
        database: str | None = None,
        schema: str | None = None,
    ) -> ir.Table:
        """Create a table expression from a table in the database.

        Parameters
        ----------
        name
            Table name
        database
            The database the table resides in
        schema
            The schema inside `database` where the table resides.

            !!! warning "`schema` refers to database organization"

                The `schema` parameter does **not** refer to the column names
                and types of `table`.

        Returns
        -------
        Table
            Table expression
        """
        if database is not None and database != self.current_database:
            return self.database(database=database).table(
                name=name,
                database=database,
                schema=schema,
            )
        sqla_table = self._get_sqla_table(
            name,
            database=database,
            schema=schema,
        )
        return self._sqla_table_to_expr(sqla_table)

    def insert(
        self,
        table_name: str,
        obj: pd.DataFrame | ir.Table,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Insert data into a table.

        Parameters
        ----------
        table_name
            The name of the table to which data needs will be inserted
        obj
            The source data or expression to insert
        database
            Name of the attached database that the table is located in.
        overwrite
            If `True` then replace existing contents of table

        Raises
        ------
        NotImplementedError
            If inserting data from a different database
        ValueError
            If the type of `obj` isn't supported
        """

        if database == self.current_database:
            # avoid fully qualified name
            database = None

        if database is not None:
            raise NotImplementedError(
                'Inserting data to a table from a different database is not '
                'yet implemented'
            )

        if isinstance(obj, pd.DataFrame):
            obj.to_sql(
                table_name,
                self.con,
                index=False,
                if_exists='replace' if overwrite else 'append',
                schema=self._current_schema,
            )
        elif isinstance(obj, ir.Table):
            to_table_expr = self.table(table_name)
            to_table_schema = to_table_expr.schema()

            if overwrite:
                self.drop_table(table_name, database=database)
                self.create_table(
                    table_name,
                    schema=to_table_schema,
                    database=database,
                )

            to_table = self._get_sqla_table(table_name, schema=database)

            from_table_expr = obj

            with self.begin() as bind:
                if from_table_expr is not None:
                    bind.execute(
                        to_table.insert().from_select(
                            list(from_table_expr.columns),
                            from_table_expr.compile(),
                        )
                    )
        else:
            raise ValueError(
                "No operation is being performed. Either the obj parameter "
                "is not a pandas DataFrame or is not a ibis Table."
                f"The given obj is of type {type(obj).__name__} ."
            )

    def _get_temp_view_definition(
        self,
        name: str,
        definition: sa.sql.compiler.Compiled,
    ) -> str:
        raise NotImplementedError(
            f"The {self.name} backend does not implement temporary view "
            "creation"
        )

    def _register_temp_view_cleanup(self, name: str, raw_name: str) -> None:
        pass

    def _create_temp_view(
        self,
        view: sa.Table,
        definition: sa.sql.Selectable,
    ) -> None:
        raw_name = view.name
        if raw_name not in self._temp_views and raw_name in self.list_tables():
            raise ValueError(f"{raw_name} already exists as a table or view")

        name = self.con.dialect.identifier_preparer.quote_identifier(raw_name)
        compiled = definition.compile()
        defn = self._get_temp_view_definition(name, definition=compiled)
        query = sa.text(defn).bindparams(**compiled.params)
        self.con.execute(query)
        self._temp_views.add(raw_name)
        self._register_temp_view_cleanup(name, raw_name)
