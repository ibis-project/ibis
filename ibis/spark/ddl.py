import re

from ibis.impala.compiler import quote_identifier
from ibis.spark.compiler import _type_to_sql_string
from ibis.sql.compiler import DDL, DML

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


def _is_fully_qualified(x):
    return bool(fully_qualified_re.search(x))


def _is_quoted(x):
    regex = re.compile(r"(?:`(.*)`|(.*))")
    quoted, _ = regex.match(x).groups()
    return quoted is not None


class SparkQualifiedSQLStatement:
    def _get_scoped_name(self, obj_name, database):
        if database:
            scoped_name = '{}.`{}`'.format(database, obj_name)
        else:
            if not _is_fully_qualified(obj_name):
                if _is_quoted(obj_name):
                    return obj_name
                else:
                    return '`{}`'.format(obj_name)
            else:
                return obj_name
        return scoped_name


class SparkDDL(DDL, SparkQualifiedSQLStatement):
    pass


class SparkDML(DML, SparkQualifiedSQLStatement):
    pass


class CreateDDL(SparkDDL):
    def _if_exists(self):
        return 'IF NOT EXISTS ' if self.can_exist else ''


_format_aliases = {'TEXTFILE': 'TEXT'}


def _sanitize_format(format):
    if format is None:
        return
    format = format.upper()
    format = _format_aliases.get(format, format)
    if format not in (
        'TEXT',
        'CSV',
        'JSON',
        'JDBC',
        'PARQUET',
        'ORC',
        'HIVE',
        'DELTA',
        'LIBSVM',
    ):
        raise ValueError('Invalid format: {!r}'.format(format))

    return format


def format_tblproperties(props):
    formatted_props = _format_properties(props)
    return 'TBLPROPERTIES {}'.format(formatted_props)


def _format_properties(props):
    tokens = []
    for k, v in sorted(props.items()):
        tokens.append("  '{}'='{}'".format(k, v))

    return '(\n{}\n)'.format(',\n'.join(tokens))


class CreateTable(CreateDDL):

    """

    Parameters
    ----------
    partition :

    """

    def __init__(
        self,
        table_name,
        database=None,
        format='parquet',
        can_exist=False,
        path=None,
        tbl_properties=None,
    ):
        self.table_name = table_name
        self.database = database
        self.path = path
        self.can_exist = can_exist
        self.format = _sanitize_format(format)
        self.tbl_properties = tbl_properties

    @property
    def _prefix(self):
        return 'CREATE TABLE'

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return '{} {}{}'.format(self._prefix, self._if_exists(), scoped_name)

    def _location(self):
        return "LOCATION '{}'".format(self.path) if self.path else None

    def _storage(self):
        # By the time we're here, we have a valid format
        return 'USING {}'.format(self.format)

    @property
    def pieces(self):
        yield self._create_line()
        for piece in filter(None, self._pieces):
            yield piece

    def compile(self):
        return '\n'.join(self.pieces)


class CreateTableWithSchema(CreateTable):
    def __init__(self, table_name, schema, table_format=None, **kwargs):
        super().__init__(table_name, **kwargs)
        self.schema = schema
        self.table_format = table_format

    @property
    def _pieces(self):
        yield format_schema(self.schema)

        if self.table_format is not None:
            yield '\n'.join(self.table_format.to_ddl())
        else:
            yield self._storage()

        yield self._location()


class CTAS(CreateTable):

    """
    Create Table As Select
    """

    def __init__(
        self,
        table_name,
        select,
        database=None,
        format='parquet',
        can_exist=False,
        path=None,
    ):
        super().__init__(
            table_name,
            database=database,
            format=format,
            can_exist=can_exist,
            path=path,
        )
        self.select = select

    @property
    def _pieces(self):
        yield self._storage()
        yield self._location()
        yield 'AS'
        yield self.select.compile()


class CreateView(CTAS):

    """Create a view"""

    def __init__(
        self,
        table_name,
        select,
        database=None,
        or_replace=True,
        temporary=False,
    ):
        super().__init__(
            table_name,
            select,
            database=database,
        )
        self.or_replace = or_replace
        self.temporary = temporary

    @property
    def _pieces(self):
        yield 'AS'
        yield self.select.compile()

    @property
    def _prefix(self):
        return 'CREATE {}{}VIEW'.format(
            self._or_replace_clause(),
            self._view_clause()
        )

    def _or_replace_clause(self):
        return 'OR REPLACE ' if self.or_replace else ''

    def _view_clause(self):
        if self.temporary:
            return 'TEMPORARY '
        return ''


def format_schema(schema):
    elements = [
        _format_schema_element(name, t)
        for name, t in zip(schema.names, schema.types)
    ]
    return '({})'.format(',\n '.join(elements))


def _format_schema_element(name, t):
    return '{} {}'.format(
        quote_identifier(name, force=True), _type_to_sql_string(t)
    )


class CreateDatabase(CreateDDL):
    def __init__(self, name, path=None, can_exist=False):
        self.name = name
        self.path = path
        self.can_exist = can_exist

    def compile(self):
        name = quote_identifier(self.name)

        create_decl = 'CREATE DATABASE'
        create_line = '{} {}{}'.format(create_decl, self._if_exists(), name)
        if self.path is not None:
            create_line += "\nLOCATION '{}'".format(self.path)

        return create_line


class DropObject(SparkDDL):
    def __init__(self, must_exist=True):
        self.must_exist = must_exist

    def compile(self):
        if_exists = '' if self.must_exist else 'IF EXISTS '
        object_name = self._object_name()
        return 'DROP {} {}{}'.format(self._object_type, if_exists, object_name)


class DropTable(DropObject):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.table_name = table_name
        self.database = database

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database)


class DropDatabase(DropObject):

    _object_type = 'DATABASE'

    def __init__(self, name, must_exist=True, cascade=False):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.cascade = cascade

    def compile(self):
        if self.cascade:
            return '{} CASCADE'.format(super().compile())
        else:
            return super().compile()

    def _object_name(self):
        return self.name


class TruncateTable(SparkDDL):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self._get_scoped_name(self.table_name, self.database)
        return 'TRUNCATE TABLE {}'.format(name)


class InsertSelect(SparkDML):
    def __init__(
        self,
        table_name,
        select_expr,
        database=None,
        overwrite=False,
    ):
        self.table_name = table_name
        self.database = database
        self.select = select_expr

        self.overwrite = overwrite

    def compile(self):
        if self.overwrite:
            cmd = 'INSERT OVERWRITE TABLE'
        else:
            cmd = 'INSERT INTO'

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return '{0} {1}\n{2}'.format(
            cmd, scoped_name, select_query
        )


class AlterTable(SparkDDL):
    def __init__(
        self,
        table,
        location=None,
        format=None,
        tbl_properties=None,
    ):
        self.table = table
        self.location = location
        self.format = _sanitize_format(format)
        self.tbl_properties = tbl_properties

    def _wrap_command(self, cmd):
        return 'ALTER TABLE {}'.format(cmd)

    def _format_properties(self, prefix=''):
        tokens = []

        if self.location is not None:
            tokens.append("LOCATION '{}'".format(self.location))

        if self.format is not None:
            tokens.append("FILEFORMAT {}".format(self.format))

        if self.tbl_properties is not None:
            tokens.append(format_tblproperties(self.tbl_properties))

        if len(tokens) > 0:
            return '\n{}{}'.format(prefix, '\n'.join(tokens))
        else:
            return ''

    def compile(self):
        props = self._format_properties()
        action = '{} SET {}'.format(self.table, props)
        return self._wrap_command(action)


class RenameTable(AlterTable):
    def __init__(
        self, old_name, new_name
    ):
        self.old_name = old_name
        self.new_name = new_name

    def compile(self):
        cmd = '{} RENAME TO {}'.format(
            self.old_name, self.new_name
        )
        return self._wrap_command(cmd)
