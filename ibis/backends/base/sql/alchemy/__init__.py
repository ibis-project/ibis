from __future__ import annotations

import abc
import atexit
import contextlib
import getpass
import warnings
from operator import methodcaller
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import sqlalchemy as sa

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.base.sql.alchemy.database import AlchemyDatabase, AlchemyTable
from ibis.backends.base.sql.alchemy.datatypes import schema_from_table, to_sqla_type
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.backends.base.sql.alchemy.query_builder import AlchemyCompiler
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    get_sqla_table,
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

if TYPE_CHECKING:
    import pandas as pd

    import ibis.expr.datatypes as dt

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
    'varargs',
)


class BaseAlchemyBackend(BaseSQLBackend):
    """Backend class for backends that compile to SQLAlchemy expressions."""

    database_class = AlchemyDatabase
    table_class = AlchemyTable
    compiler = AlchemyCompiler

    def _build_alchemy_url(self, url, host, port, user, password, database, driver):
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

    def _to_sql(self, expr: ir.Expr, **kwargs) -> str:
        # For `ibis.to_sql` calls we render with literal binds and qmark params
        dialect_class = sa.dialects.registry.load(
            self.compiler.translator_class._dialect_name
        )
        sql = self.compile(expr, **kwargs).compile(
            dialect=dialect_class(paramstyle="qmark"),
            compile_kwargs=dict(literal_binds=True),
        )
        return str(sql)

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with self.begin() as con:
            yield con.execute(*args, **kwargs)

    @staticmethod
    def _to_geodataframe(df, schema):
        """Convert `df` to a `GeoDataFrame`.

        Required libraries for geospatial support must be installed and
        a geospatial column is present in the dataframe.
        """
        import geopandas as gpd
        from geoalchemy2 import shape

        geom_col = None
        for name, dtype in schema.items():
            if dtype.is_geospatial():
                geom_col = geom_col or name
                df[name] = df[name].map(
                    lambda row: None if row is None else shape.to_shape(row)
                )
        if geom_col:
            df[geom_col] = gpd.array.GeometryArray(df[geom_col].values)
            df = gpd.GeoDataFrame(df, geometry=geom_col)
        return df

    def fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        df = pd.DataFrame.from_records(cursor, columns=schema.names, coerce_float=True)
        df = schema.apply_to(df)
        if not df.empty and geospatial_supported:
            return self._to_geodataframe(df, schema)
        return df

    @contextlib.contextmanager
    def begin(self):
        with self.con.begin() as bind:
            yield bind

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table.

        Parameters
        ----------
        name
            Name of the new table.
        obj
            An Ibis table expression or pandas table that will be used to
            extract the schema and the data of the new table. If not provided,
            `schema` must be given.
        schema
            The schema for the new table. Only one of `schema` or `obj` can be
            provided.
        database
            Name of the database where the table will be created, if not the
            default.
        temp
            Should the table be temporary for the session.
        overwrite
            Clobber existing data

        Returns
        -------
        Table
            The table that was created.
        """
        if obj is None and schema is None:
            raise com.IbisError("The schema or obj parameter is required")

        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            obj = ibis.memtable(obj)

        if database == self.current_database:
            # avoid fully qualified name
            database = None

        if database is not None:
            raise NotImplementedError(
                "Creating tables from a different database is not yet implemented"
            )

        if obj is not None and schema is not None:
            if not obj.schema().equals(ibis.schema(schema)):
                raise com.IbisTypeError(
                    'Expression schema is not equal to passed schema. '
                    'Try passing the expression without the schema'
                )
        if schema is None:
            schema = obj.schema()

        self._schemas[self._fully_qualified_name(name, database)] = schema
        table = self._table_from_schema(
            name,
            schema,
            database=database or self.current_database,
            temp=temp,
        )

        if has_expr := obj is not None:
            # this has to happen outside the `begin` block, so that in-memory
            # tables are visible inside the transaction created by it
            self._register_in_memory_tables(obj)

        with self.begin() as bind:
            if overwrite:
                table.drop(bind=bind, checkfirst=True)
            table.create(bind=bind)
            if has_expr:
                method = self._get_insert_method(obj)
                bind.execute(method(table.insert()))
        return self.table(name, database=database)

    def _get_insert_method(self, expr):
        compiled = self.compile(expr)

        # if in memory tables aren't cheap then try to pull out their data
        # FIXME: queries that *select* from in memory tables are still broken
        # for mysql/sqlite/postgres because the generated SQL is wrong
        if (
            not self.compiler.cheap_in_memory_tables
            and self.compiler.support_values_syntax_in_select
            and isinstance(expr.op(), ops.InMemoryTable)
        ):
            (from_,) = compiled.get_final_froms()
            try:
                (rows,) = from_._data
            except AttributeError:
                return methodcaller("from_select", list(expr.columns), from_)
            else:
                return methodcaller("values", rows)

        return methodcaller("from_select", list(expr.columns), compiled)

    def _columns_from_schema(self, name: str, schema: sch.Schema) -> list[sa.Column]:
        dialect = self.con.dialect
        return [
            sa.Column(
                colname,
                to_sqla_type(dialect, dtype),
                nullable=dtype.nullable,
                quote=self.compiler.translator_class._quote_column_names,
            )
            for colname, dtype in zip(schema.names, schema.types)
        ]

    def _table_from_schema(
        self, name: str, schema: sch.Schema, temp: bool = False, **_: Any
    ) -> sa.Table:
        prefixes = []
        if temp:
            prefixes.append('TEMPORARY')
        columns = self._columns_from_schema(name, schema)
        return sa.Table(
            name,
            sa.MetaData(),
            *columns,
            prefixes=prefixes,
            quote=self.compiler.translator_class._quote_table_names,
        )

    def drop_table(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        """Drop a table.

        Parameters
        ----------
        name
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
            raise com.IbisInputError(
                "Dropping tables from a different database is not yet implemented"
            )

        t = self._get_sqla_table(name, schema=database, autoload=False)
        with self.begin() as bind:
            t.drop(bind=bind, checkfirst=force)

        qualified_name = self._fully_qualified_name(name, database)

        with contextlib.suppress(KeyError):
            # schemas won't be cached if created with raw_sql
            del self._schemas[qualified_name]

    @util.deprecated(
        as_of="5.0", removed_in="6.0", instead="Use create_table(overwrite=True)"
    )
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

    def truncate_table(self, name: str, database: str | None = None) -> None:
        t = self._get_sqla_table(name, schema=database)
        with self.begin() as con:
            con.execute(t.delete())

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

    def _log(self, sql):
        try:
            query_str = str(sql)
        except sa.exc.UnsupportedCompilationError:
            pass
        else:
            util.log(query_str)

    def _get_sqla_table(
        self, name: str, schema: str | None = None, autoload: bool = True, **_: Any
    ) -> sa.Table:
        # If the underlying table (or more likely, view) has changed, remove it
        # to ensure a correct reflection
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Did not recognize type", category=sa.exc.SAWarning
            )
            table = sa.Table(
                name,
                sa.MetaData(),
                schema=schema,
                autoload_with=self.con if autoload else None,
                quote=self.compiler.translator_class._quote_table_names,
            )
            nulltype_cols = frozenset(
                col.name for col in table.c if isinstance(col.type, sa.types.NullType)
            )

            if not nulltype_cols:
                return table
            return self._handle_failed_column_type_inference(table, nulltype_cols)

    def _handle_failed_column_type_inference(
        self, table: sa.Table, nulltype_cols: Iterable[str]
    ) -> sa.Table:
        """Handle cases where SQLAlchemy cannot infer the column types of `table`."""

        self.inspector.reflect_table(table, table.columns)
        dialect = self.con.dialect
        quoted_name = dialect.identifier_preparer.quote(table.name)

        for colname, type in self._metadata(quoted_name):
            if colname in nulltype_cols:
                # replace null types discovered by sqlalchemy with non null
                # types
                table.append_column(
                    sa.Column(
                        colname,
                        to_sqla_type(dialect, type),
                        nullable=type.nullable,
                        quote=self.compiler.translator_class._quote_column_names,
                    ),
                    replace_existing=True,
                )
        return table

    def _sqla_table_to_expr(self, table: sa.Table) -> ir.Table:
        schema = self._schemas.get(table.name)
        node = self.table_class(
            source=self,
            sqla_table=table,
            name=table.name,
            schema=schema,
        )
        return self.table_expr_class(node)

    def raw_sql(self, query) -> None:
        """Execute a query string.

        !!! warning "The returned cursor object must be **manually** released."

        Parameters
        ----------
        query
            DDL or DML statement
        """
        return self.con.connect().execute(
            sa.text(query) if isinstance(query, str) else query
        )

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
        if database is not None:
            if not isinstance(database, str):
                raise com.IbisTypeError(
                    f"`database` must be a string; got {type(database)}"
                )
            if database != self.current_database:
                return self.database(name=database).table(name=name, schema=schema)
        sqla_table = self._get_sqla_table(name, database=database, schema=schema)
        return self._sqla_table_to_expr(sqla_table)

    def _insert_dataframe(
        self, table_name: str, df: pd.DataFrame, overwrite: bool
    ) -> None:
        schema = self._current_schema

        t = self._get_sqla_table(table_name, schema=schema)
        with self.con.begin() as con:
            if overwrite:
                con.execute(t.delete())
            con.execute(t.insert(), df.to_dict(orient="records"))

    def insert(
        self,
        table_name: str,
        obj: pd.DataFrame | ir.Table | list | dict,
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

        import pandas as pd

        if database == self.current_database:
            # avoid fully qualified name
            database = None

        if database is not None:
            raise NotImplementedError(
                'Inserting data to a table from a different database is not '
                'yet implemented'
            )

        # If we've been passed a `memtable`, pull out the underlying dataframe
        if isinstance(obj, ir.Table) and isinstance(
            in_mem_table := obj.op(), ops.InMemoryTable
        ):
            obj = in_mem_table.data.to_frame()

        if isinstance(obj, pd.DataFrame):
            self._insert_dataframe(table_name, obj, overwrite=overwrite)
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
                    compiled = from_table_expr.compile()
                    columns = [
                        self.con.dialect.normalize_name(c)
                        for c in from_table_expr.columns
                    ]
                    bind.execute(to_table.insert().from_select(columns, compiled))
        elif isinstance(obj, (list, dict)):
            to_table = self._get_sqla_table(table_name, schema=database)

            with self.begin() as bind:
                if overwrite:
                    bind.execute(to_table.delete())
                bind.execute(to_table.insert().values(obj))

        else:
            raise ValueError(
                "No operation is being performed. Either the obj parameter "
                "is not a pandas DataFrame or is not a ibis Table."
                f"The given obj is of type {type(obj).__name__} ."
            )

    def _quote(self, name: str) -> str:
        """Quote an identifier."""
        preparer = self.con.dialect.identifier_preparer
        if self.compiler.translator_class._quote_table_names:
            return preparer.quote_identifier(name)
        return preparer.quote(name)

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        raise NotImplementedError(
            f"The {self.name} backend does not implement temporary view creation"
        )

    def _register_temp_view_cleanup(self, name: str, raw_name: str) -> None:
        query = f"DROP VIEW IF EXISTS {name}"

        def drop(self, raw_name: str, query: str):
            with self.begin() as con:
                con.exec_driver_sql(query)
            self._temp_views.discard(raw_name)

        atexit.register(drop, self, raw_name, query)

    def _get_compiled_statement(
        self,
        definition: sa.sql.Selectable,
        name: str,
        compile_kwargs: Mapping[str, Any] | None = None,
    ):
        if compile_kwargs is None:
            compile_kwargs = {}
        compiled = definition.compile(
            dialect=self.con.dialect, compile_kwargs=compile_kwargs
        )
        lines = self._get_temp_view_definition(name, definition=compiled)
        return lines, compiled.params

    def _create_temp_view(self, view: sa.Table, definition: sa.sql.Selectable) -> None:
        raw_name = view.name
        if raw_name not in self._temp_views and raw_name in self.list_tables():
            raise ValueError(f"{raw_name} already exists as a table or view")
        name = self._quote(raw_name)
        lines, params = self._get_compiled_statement(definition, name)
        with self.begin() as con:
            for line in lines:
                con.exec_driver_sql(line, parameters=params or ())
        self._temp_views.add(raw_name)
        self._register_temp_view_cleanup(name, raw_name)

    @abc.abstractmethod
    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        ...

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a backend-specific SQL string."""
        return sch.Schema.from_tuples(self._metadata(query))

    def _load_into_cache(self, name, expr):
        self.create_table(name, expr, schema=expr.schema(), temp=True)

    def _clean_up_cached_table(self, op):
        self.drop_table(op.name)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        import sqlalchemy_views as sav

        source = self.compile(obj)
        view = sav.CreateView(
            sa.Table(
                name,
                sa.MetaData(),
                schema=database,
                quote=self.compiler.translator_class._quote_table_names,
            ),
            source,
            or_replace=overwrite,
        )
        with self.begin() as con:
            con.execute(view)
        return self.table(name, database=database)

    def drop_view(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        import sqlalchemy_views as sav

        view = sav.DropView(
            sa.Table(
                name,
                sa.MetaData(),
                schema=database,
                quote=self.compiler.translator_class._quote_table_names,
            ),
            if_exists=not force,
        )

        with self.begin() as con:
            con.execute(view)
