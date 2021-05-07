"""Module for DDL operations."""
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ibis
from ibis.backends.base_sqlalchemy.compiler import DDL, DML
from ibis.common import exceptions as com

from . import dtypes as omniscidb_dtypes
from .compiler import _type_to_sql_string, quote_identifier

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


def _is_fully_qualified(x):
    return bool(fully_qualified_re.search(x))


def _is_quoted(x):
    regex = re.compile(r"(?:`(.*)`|(.*))")
    quoted, _ = regex.match(x).groups()
    return quoted is not None


def _convert_default_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "'t'" if value else "'f'"
    if isinstance(value, (int, float)):
        return value
    return quote_identifier(value, force=True)


def _bool2str(v: bool) -> str:
    """
    Convert a bool value to a OmniSciDB bool value.

    Parameters
    ----------
    v : bool

    Returns
    -------
    str
    """
    return str(bool(v)).lower()


class OmniSciDBQualifiedSQLStatement:
    """OmniSciDBQualifiedSQLStatement."""

    def _get_scoped_name(self, obj_name, database):  # noqa: F401
        return obj_name


class OmniSciDBDDL(DDL, OmniSciDBQualifiedSQLStatement):
    """OmniSciDB DDL class."""


class OmniSciDBDML(DML, OmniSciDBQualifiedSQLStatement):
    """OmniSciDB DML class."""


class CreateDDL(OmniSciDBDDL):
    """Create DDL."""


class DropObject(OmniSciDBDDL):
    """Drop object class."""

    def __init__(self, must_exist=True):
        """Initialize the drop object operation."""
        self.must_exist = must_exist

    def compile(self):
        """Compile the drop object operation."""
        if_exists = '' if self.must_exist else 'IF EXISTS '
        object_name = self._object_name()
        return 'DROP {} {}{}'.format(self._object_type, if_exists, object_name)


class DropTable(DropObject):
    """Drop table class."""

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None, must_exist=True):
        """Initialize the drop table object."""
        super().__init__(must_exist=must_exist)
        self.table_name = table_name
        self.database = database

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database)


class CreateTable(CreateDDL):
    """Create Table class.

    Parameters
    ----------
    table_name : string
    database : string
    """

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    @property
    def _prefix(self):
        return 'CREATE TABLE'

    def _create_line(self):
        return '{} {}'.format(self._prefix, self.table_name)

    @property
    def pieces(self):
        """Get all the pieces of the Create Table expression.

        Yields
        ------
        string
        """
        yield self._create_line()
        for piece in filter(None, self._pieces):
            yield piece

    def compile(self):
        """Compile the create table expression.

        Returns
        -------
        string
        """
        return '\n'.join(self.pieces)


class CreateTableWithSchema(CreateTable):
    """Create Table With Schema class."""

    def __init__(
        self,
        table_name: str,
        schema: ibis.Schema,
        database: Optional[str] = None,
        max_rows: Optional[int] = None,
        fragment_size: Optional[int] = None,
        is_temporary: bool = False,
    ):
        """
        Initialize CreateTableWithSchema.

        Parameters
        ----------
        table_name : str
        schema : ibis.Schema
        database : str, optional, defaul None
        max_rows : int, optional, defaul None
        fragment_size : int, optional, defaul None
        is_temporary : bool, default False
        """
        self.table_name = table_name
        self.database = database
        self.schema = schema
        self.max_rows = max_rows
        self.fragment_size = fragment_size
        self.is_temporary = is_temporary

    @property
    def _prefix(self):
        return 'CREATE {}TABLE'.format(
            'TEMPORARY ' if self.is_temporary else ''
        )

    @property
    def with_params(self) -> Dict[str, Any]:
        """Return the parameters for `with` clause.

        Returns
        -------
        Dict[str, Any]
        """
        return dict(max_rows=self.max_rows, fragment_size=self.fragment_size)

    @property
    def _pieces(self):
        yield format_schema(self.schema)

        with_stmt = ','.join(
            [
                '{}={}'.format(
                    i, "'{}'".format(v) if isinstance(v, str) else v
                )
                for i, v in self.with_params.items()
                if v is not None
            ]
        )

        if with_stmt:
            yield ' WITH ({})'.format(with_stmt)


class CTAS(CreateTable):
    """Create Table As Select."""

    def __init__(self, table_name, select, database=None):
        self.table_name = table_name
        self.database = database
        self.select = select

    @property
    def _prefix(self):
        return 'CREATE TABLE'

    @property
    def _pieces(self):
        yield 'AS ('
        yield self.select.compile()
        yield ')'


# VIEW


class CreateView(CTAS):
    """Create a view."""

    def __init__(self, table_name, select, database=None):
        super().__init__(table_name, select, database=database)

    @property
    def _pieces(self):
        yield 'AS'
        yield self.select.compile()

    @property
    def _prefix(self):
        return 'CREATE VIEW'


class DropView(DropTable):
    """Drop View class."""

    _object_type = 'VIEW'


# DDL User classes


class AlterUser(OmniSciDBDDL):
    """Create user."""

    def __init__(
        self,
        name,
        password=None,
        database=None,
        is_super=False,
        insert_access=None,
    ):
        self.name = name
        self.password = password
        self.database = database
        self.is_super = is_super
        self.insert_access = insert_access

    @property
    def _params(self):
        if self.password is not None:
            yield "  password='{}'".format(self.password)

        if self.is_super is not None:
            yield "  is_super='{}'".format(
                'true' if self.is_super else 'false'
            )
        if self.insert_access:
            yield "  INSERTACCESS='{}'".format(self.insert_access)

    @property
    def pieces(self):
        """Get all the pieces for the Alter User expression."""
        yield 'ALTER USER {} ('.format(self.name)
        yield ','.join(self._params)
        yield ')'

    def compile(self):
        """Compile the Alter User expression.

        Returns
        -------
        string
        """
        return '\n'.join(self.pieces)


class CreateUser(OmniSciDBDDL):
    """Create user."""

    def __init__(self, name, password, database=None, is_super=False):
        self.name = name
        self.password = password
        self.database = database
        self.is_super = is_super

    @property
    def pieces(self):
        """Get all the pieces for the Create User expression.

        Yields
        ------
        string
        """
        yield 'CREATE USER {} ('.format(self.name)
        yield "  password='{}',".format(self.password)
        yield "  is_super='{}'".format('true' if self.is_super else 'false')
        yield ')'

    def compile(self):
        """Compile the Create User expression.

        Returns
        -------
        string
        """
        return '\n'.join(self.pieces)


class DropUser(OmniSciDBDDL):
    """Drop user."""

    def __init__(self, name, database=None):
        self.name = name
        self.database = database

    @property
    def pieces(self):
        """Get all the pieces for the Drop User expression.

        Yields
        ------
        string
        """
        yield 'DROP USER {}'.format(self.name)

    def compile(self):
        """Compile the Drop User expression.

        Returns
        -------
        string
        """
        return '\n'.join(self.pieces)


# DDL Table classes


class AlterTable(OmniSciDBDDL):
    """Alter Table class."""

    def __init__(self, args, **kwargs):
        raise NotImplementedError('Not implemented yet.')

    def _wrap_command(self, cmd):
        return 'ALTER TABLE {}'.format(cmd)


class AddColumns(AlterTable):
    """Add Columns class."""

    def __init__(
        self,
        table_name: str,
        cols_with_types: dict,
        nullables: Optional[list] = None,
        defaults: Optional[list] = None,
        encodings: Optional[list] = None,
    ):
        if len(cols_with_types) == 0:
            raise com.IbisInputError('No column requested to add.')
        else:
            self.col_count = len(cols_with_types)
        self.table_name = table_name
        self.cols_with_types = cols_with_types

        if not nullables:
            self.nullables = [True] * self.col_count
        else:
            self.nullables = nullables

        if not defaults:
            self.defaults = [None] * self.col_count
        else:
            self.defaults = defaults

        if not encodings:
            self.encodings = [None] * self.col_count
        else:
            self.encodings = encodings

    def _pieces(self):
        idx = 0
        sep = ''
        yield '{} ADD ('.format(self.table_name)
        for col, d_type in self.cols_with_types.items():
            yield '{}{} {}{}{}{}'.format(
                sep,
                col,
                omniscidb_dtypes.ibis_dtypes_str_to_sql[d_type],
                ' NOT NULL'
                if not self.nullables[idx] and self.defaults[idx] is None
                else '',
                ' DEFAULT {}'.format(
                    _convert_default_value(self.defaults[idx])
                )
                if self.defaults[idx] is not None
                else '',
                ' ENCODING {}'.format(self.encodings[idx])
                if self.encodings[idx]
                else '',
            )
            idx += 1
            sep = ', '
        yield ');'

    def compile(self):
        """Compile the Add Column expression.

        Returns
        -------
        string
        """
        cmd = "".join(self._pieces())
        return self._wrap_command(cmd)


class DropColumns(AlterTable):
    """Drop Columns class."""

    def __init__(self, table_name: str, column_names: list):
        if len(column_names) == 0:
            raise com.IbisInputError('No column requested to drop.')
        self.table_name = table_name
        self.column_names = column_names

    def _pieces(self):
        sep = ''
        yield '{}'.format(self.table_name)
        for col in self.column_names:
            yield '{} DROP {}'.format(sep, col)
            sep = ','
        yield ';'

    def compile(self):
        """Compile the Drop Column expression.

        Returns
        -------
        string
        """
        cmd = "".join(self._pieces())
        return self._wrap_command(cmd)


class RenameTable(AlterTable):
    """Rename Table class."""

    def __init__(
        self, old_name, new_name, old_database=None, new_database=None
    ):
        # if either database is None, the name is assumed to be fully scoped
        self.old_name = old_name
        self.old_database = old_database
        self.new_name = new_name
        self.new_database = new_database

        new_qualified_name = new_name
        if new_database is not None:
            new_qualified_name = self._get_scoped_name(new_name, new_database)

        old_qualified_name = old_name
        if old_database is not None:
            old_qualified_name = self._get_scoped_name(old_name, old_database)

        self.old_qualified_name = old_qualified_name
        self.new_qualified_name = new_qualified_name

    def compile(self):
        """Compile the Rename Table expression.

        Returns
        -------
        string
        """
        cmd = '{} RENAME TO {}'.format(
            self.old_qualified_name, self.new_qualified_name
        )
        return self._wrap_command(cmd)


class TruncateTable(OmniSciDBDDL):
    """Truncate Table class."""

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        """Compile Truncate Table class.

        Returns
        -------
        string
        """
        name = self._get_scoped_name(self.table_name, self.database)
        return 'TRUNCATE TABLE {}'.format(name)


class CacheTable(OmniSciDBDDL):
    """Cache Table class."""

    def __init__(self, table_name, database=None, pool='default'):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self):
        """Compile Cache Table class.

        Returns
        -------
        string
        """
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return "ALTER TABLE {} SET CACHED IN '{}'".format(
            scoped_name, self.pool
        )


class CreateDatabase(CreateDDL):
    """Create Database class."""

    def __init__(self, name, owner=None):
        self.name = name
        self.owner = owner

    def compile(self):
        """Compile Create Database expression.

        Returns
        -------
        string
        """
        name = quote_identifier(self.name)

        cmd = 'CREATE DATABASE'
        properties = ''

        if self.owner:
            properties = '(owner=\'{}\')'.format(self.owner)

        return '{} {} {}'.format(cmd, name, properties)


class DropDatabase(DropObject):
    """Drop Database class."""

    _object_type = 'DATABASE'

    def __init__(self, name):
        super().__init__(must_exist=True)
        self.name = name

    def _object_name(self):
        return self.name


def format_schema(schema: ibis.expr.schema.Schema):
    """Get a formatted string for a given schema.

    Parameters
    ----------
    schema : ibis.expr.schema.Schema

    Returns
    -------
    string
    """
    elements = [
        _format_schema_element(name, tp, nullable)
        for name, tp, nullable in zip(
            schema.names, schema.types, [t.nullable for t in schema.types]
        )
    ]
    return '({})'.format(',\n '.join(elements))


def _format_schema_element(name, tp, nullable):
    return '{} {} {}'.format(
        quote_identifier(name, force=False),
        _type_to_sql_string(tp),
        'NOT NULL' if not nullable else '',
    )


class InsertPandas(OmniSciDBDML):
    """Insert Data from Pandas class."""

    def __init__(self, table_name, df, insert_index=False, database=None):
        self.table_name = table_name
        self.database = database
        self.df = df.copy()

        if insert_index:
            self.df.reset_index(inplace=True)

    def _get_field_names(self):
        return ','.join(self.df.columns)

    def _get_value(self, v):
        if isinstance(v, str):
            return "'{}'".format(v)
        elif v is None:
            return 'NULL'
        else:
            return '{}'.format(v)

    def _get_field_values(self):
        for i, row in self.df[self.df.columns].iterrows():
            yield [self._get_value(v) for v in row]

    @property
    def pieces(self):
        """Get all the pieces for the Insert expression.

        Yields
        ------
        string
        """
        cmd = 'INSERT INTO'

        fields = self._get_field_names()

        stmt = '{0} {1} ({2}) VALUES '.format(cmd, self.table_name, fields)

        for values in self._get_field_values():
            yield '{} ({});'.format(stmt, ','.join(values))

    def compile(self):
        """Compile the Insert expression."""
        return '\n'.join(self.pieces)


class LoadData(OmniSciDBDDL):
    """Generate DDL for LOAD DATA command. Cannot be cancelled."""

    def __init__(
        self,
        table_name: str,
        source: Union[str, Path],
        **kwargs: Dict[str, Any],
    ):
        self.table_name = table_name
        self.source = source
        self.options = kwargs

    def _get_options(self) -> str:
        with_stmt = ','.join(
            [
                '{}={}'.format(
                    i, "'{}'".format(v) if isinstance(v, (str, bool)) else v
                )
                for i, v in self.options.items()
                if v is not None
            ]
        )
        return ' WITH ({})'.format(with_stmt)

    def compile(self) -> str:
        """Compile the LoadData expression."""
        return "COPY {} FROM '{}' {}".format(
            self.table_name, self.source, self._get_options()
        )
