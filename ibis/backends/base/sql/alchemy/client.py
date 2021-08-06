import contextlib
import warnings
from typing import Dict, List, Optional, Union

import pandas as pd
import sqlalchemy as sa

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base.sql import SQLClient

from .datatypes import to_sqla_type
from .geospatial import geospatial_supported
from .query_builder import AlchemyCompiler


class _AutoCloseCursor:
    """
    Wraps a SQLAlchemy ResultProxy and ensures that .close() is called on
    garbage collection
    """

    def __init__(self, original_cursor):
        self.original_cursor = original_cursor

    def __del__(self):
        self.original_cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.original_cursor.close()

    def fetchall(self):
        return self.original_cursor.fetchall()


class AlchemyClient(SQLClient):

    compiler = AlchemyCompiler
    has_attachment = False

    def __init__(self, con: sa.engine.Engine) -> None:
        super().__init__()
        self.con = con
        self.meta = sa.MetaData(bind=con)
        self._inspector = sa.inspect(con)
        self._schemas: Dict[str, sch.Schema] = {}

    @property
    def inspector(self):
        self._inspector.info_cache.clear()
        return self._inspector

    @staticmethod
    def _to_geodataframe(df, schema):
        """
        If the required libraries for geospatial support are installed, and if
        a geospatial column is present in the dataframe, convert it to a
        GeoDataFrame.
        """
        import geopandas
        from geoalchemy2 import shape

        def to_shapely(row, name):
            return shape.to_shape(row[name]) if row[name] is not None else None

        geom_col = None
        for name, dtype in schema.items():
            if isinstance(dtype, dt.GeoSpatial):
                geom_col = geom_col or name
                df[name] = df.apply(lambda x: to_shapely(x, name), axis=1)
        if geom_col:
            df = geopandas.GeoDataFrame(df, geometry=geom_col)
        return df

    def fetch_from_cursor(self, cursor, schema):
        df = pd.DataFrame.from_records(
            cursor.original_cursor.fetchall(),
            columns=cursor.original_cursor.keys(),
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

    def create_table(self, name, expr=None, schema=None, database=None):
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
            t.create(bind=bind)
            if expr is not None:
                bind.execute(
                    t.insert().from_select(list(expr.columns), expr.compile())
                )

    def _columns_from_schema(
        self, name: str, schema: sch.Schema
    ) -> List[sa.Column]:
        return [
            sa.Column(colname, to_sqla_type(dtype), nullable=dtype.nullable)
            for colname, dtype in zip(schema.names, schema.types)
        ]

    def _table_from_schema(
        self, name: str, schema: sch.Schema, database: Optional[str] = None
    ) -> sa.Table:
        columns = self._columns_from_schema(name, schema)
        return sa.Table(name, self.meta, *columns)

    def drop_table(
        self,
        table_name: str,
        database: Optional[str] = None,
        force: bool = False,
    ) -> None:
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

        assert (
            not t.exists()
        ), 'Something went wrong during DROP of table {!r}'.format(t.name)

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
        database: str = None,
        if_exists: str = 'fail',
    ):
        """
        Load data from a dataframe to the backend.

        Parameters
        ----------
        table_name : string
        data : pandas.DataFrame
        database : string, optional
        if_exists : string, optional, default 'fail'
            The values available are: {‘fail’, ‘replace’, ‘append’}

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

        params = {}
        if self.has_attachment:
            # for database with attachment
            # see: https://github.com/ibis-project/ibis/issues/1930
            params['schema'] = self.current_database

        data.to_sql(
            table_name,
            con=self.con,
            index=False,
            if_exists=if_exists,
            **params,
        )

    def truncate_table(
        self, table_name: str, database: Optional[str] = None
    ) -> None:
        t = self._get_sqla_table(table_name, schema=database)
        t.delete().execute()

    def schema(self, name):
        """Get a schema object from the current database for the schema named `name`.

        Parameters
        ----------
        name : str

        Returns
        -------
        schema : ibis Schema
            The schema of the object `name`.

        """
        return self.database().schema(name)

    @property
    def current_database(self):
        """The name of the current database this client is connected to."""
        return self.database_name

    def set_database(self, name):
        raise NotImplementedError(
            f'Cannot set database with {self.__class__.__name__} client. '
            f'To use a different database, use `client.database("{name}")`'
        )

    def list_schemas(self):
        warnings.warn(
            '`list_schemas` is deprecated, use `list_databases` instead',
            FutureWarning,
        )
        return self.list_databases()

    def list_databases(self):
        """List databases in the current server."""
        return self.inspector.get_schema_names()

    def list_tables(
        self,
        like: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> List[str]:
        """List tables/views in the current or indicated database.

        Parameters
        ----------
        like
            Checks for this string contained in name
        database
            If not passed, uses the current database
        schema
            The schema namespace that tables should be listed from

        Returns
        -------
        List[str]

        """
        inspector = self.inspector
        # inspector returns a mutable version of its names, so make a copy.
        names = inspector.get_table_names(schema=schema).copy()
        names.extend(inspector.get_view_names(schema=schema))
        if like is not None:
            names = [x for x in names if like in x]
        return sorted(names)

    def raw_sql(self, query: str, results=False):
        return _AutoCloseCursor(super().raw_sql(query))

    def _log(self, sql):
        try:
            query_str = str(sql)
        except sa.exc.UnsupportedCompilationError:
            pass
        else:
            util.log(query_str)

    def _get_sqla_table(self, name, schema=None, autoload=True):
        return sa.Table(name, self.meta, schema=schema, autoload=autoload)

    def _sqla_table_to_expr(self, table):
        node = self.table_class(table, self)
        return self.table_expr_class(node)

    def insert(
        self,
        table_name: str,
        obj: Union[pd.DataFrame, ir.TableExpr],
        database: Optional[str] = None,
        overwrite: Optional[bool] = False,
    ) -> None:
        """
        Insert the given data to a table in backend.

        Parameters
        ----------
        table_name : string
            name of the table to which data needs to be inserted
        obj : pandas DataFrame or ibis TableExpr
            obj is either the dataframe (pd.DataFrame) containing data
            which needs to be inserted to table_name or
            the TableExpr type which ibis provides with data which needs
            to be inserted to table_name
        database : string, optional
            name of the attached database that the table is located in.
        overwrite : boolean, default False
            If True, will replace existing contents of table else not

        Raises
        -------
        NotImplementedError
            Inserting data to a table from a different database is not
            yet implemented

        ValueError
            No operation is being performed. Either the obj parameter
            is not a pandas DataFrame or is not a ibis TableExpr.
            The given obj is of type type(obj).__name__ .

        """

        if database == self.current_database:
            # avoid fully qualified name
            database = None

        if database is not None:
            raise NotImplementedError(
                'Inserting data to a table from a different database is not '
                'yet implemented'
            )

        params = {}
        if self.has_attachment:
            # for database with attachment
            # see: https://github.com/ibis-project/ibis/issues/1930
            params['schema'] = self.current_database

        if isinstance(obj, pd.DataFrame):
            obj.to_sql(
                table_name,
                self.con,
                index=False,
                if_exists='replace' if overwrite else 'append',
                **params,
            )
        elif isinstance(obj, ir.TableExpr):
            to_table_expr = self.table(table_name)
            to_table_schema = to_table_expr.schema()

            if overwrite:
                self.drop_table(table_name, database=database)
                self.create_table(
                    table_name, schema=to_table_schema, database=database,
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
                "is not a pandas DataFrame or is not a ibis TableExpr."
                f"The given obj is of type {type(obj).__name__} ."
            )
