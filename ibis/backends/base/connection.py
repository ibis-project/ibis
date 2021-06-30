from __future__ import annotations

import abc
from typing import List, Optional

import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import Database
from ibis.common.exceptions import TranslationError


class BaseConnection(abc.ABC):
    """
    Base class for all backend connections.

    All Ibis backends are expected to subclass this `Connection` class,
    and implement all the required methods.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the backend, for example 'sqlite'.
        """

    @property
    @abc.abstractmethod
    def kind(self) -> str:
        """
        Backend kind. One of:

        sqlalchemy
            Backends using a SQLAlchemy dialect.
        sql
            SQL based backends, not based on a SQLAlchemy dialect.
        pandas
            Backends using pandas to store data and perform computations.
        spark
            Spark based backends.
        """

    @abc.abstractmethod
    def connect(connection_string: str, **options) -> BaseConnection:
        """
        Connect to the underlying database and return a client object.
        """

    def register_options(self) -> None:
        """
        If the backend has custom options, register them here.
        They will be prefixed with the name of the backend.
        """

    def compile(self, expr: ir.Expr, params=None) -> str:
        """
        Compile the expression.
        """
        return self.compiler.to_sql(expr, params=params)

    def verify(self, expr: ir.Expr, params=None):
        """
        Verify `expr` is an expression that can be compiled.
        """
        try:
            self.compile(expr, params=params)
        except TranslationError:
            return False
        else:
            return True

    @property
    @abc.abstractmethod
    def version(self):
        """Version of the backend server.

        For example, for a PostgreSQL backend this could be 13.3. The returned
        object is a `Version` object of the `packaging` package."""

    @property
    def current_database(self) -> str:
        """Return the current database."""
        # TODO: If we can assume this, we should define the `con` attribute
        return self.con.database

    def database(self, name: Optional[str] = None) -> Database:
        """Create a database object.

        Create a Database object for a given database name that can be used for
        exploring and manipulating the objects (tables, functions, views, etc.)
        inside.

        Parameters
        ----------
        name : string
          Name of database

        Returns
        -------
        database : Database
        """
        # TODO: validate existence of database
        if name is None:
            name = self.current_database
        return self.database_class(name, self)

    @abc.abstractmethod
    def get_schema(
        self, table_name: str, database: Optional[str] = None
    ) -> sch.Schema:
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

    def _fully_qualified_name(self, name, database):
        """Full name of a database object, including the database."""
        # TODO this default should include the database
        return name

    def table(self, name: str, database: Optional[str] = None) -> ir.TableExpr:
        """Create a table expression.

        Create a table expression that references a particular table in the
        database.

        Parameters
        ----------
        name : string
        database : string, optional

        Returns
        -------
        table : TableExpr
        """
        qualified_name = self._fully_qualified_name(name, database)
        schema = self.get_schema(qualified_name)
        node = self.table_class(qualified_name, schema, self)
        return self.table_expr_class(node)

    @abc.abstractmethod
    def list_tables(
        self, like: Optional[str] = None, database: Optional[str] = None
    ) -> List[str]:
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
