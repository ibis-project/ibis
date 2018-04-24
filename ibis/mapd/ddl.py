from ibis.sql.compiler import DDL, DML
from .compiler import quote_identifier, _type_to_sql_string

import ibis.expr.schema as sch
import re

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


def _is_fully_qualified(x):
    return bool(fully_qualified_re.search(x))


def _is_quoted(x):
    regex = re.compile(r"(?:`(.*)`|(.*))")
    quoted, _ = regex.match(x).groups()
    return quoted is not None


class MapDQualifiedSQLStatement(object):
    def _get_scoped_name(self, obj_name, database):  # noqa: F401
        return obj_name


class MapDDDL(DDL, MapDQualifiedSQLStatement):
    pass


class MapDDML(DML, MapDQualifiedSQLStatement):
    pass


class CreateDDL(MapDDDL):

    def _if_exists(self):
        return 'IF NOT EXISTS ' if self.can_exist else ''


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
        self, table_name, database=None, can_exist=False, tbl_properties=None
    ):
        self.table_name = table_name
        self.database = database
        self.can_exist = can_exist
        self.tbl_properties = tbl_properties

    @property
    def _prefix(self):
        return 'CREATE TABLE'

    def _create_line(self):
        return '{} {}{}'.format(
            self._prefix, self._if_exists(), self.table_name
        )

    @property
    def pieces(self):
        yield self._create_line()
        for piece in filter(None, self._pieces):
            yield piece

    def compile(self):
        return '\n'.join(self.pieces)


class CTAS(CreateTable):

    """
    Create Table As Select
    """

    def __init__(
        self, table_name, select, database=None, can_exist=False,
    ):
        super(CTAS, self).__init__(
            table_name, database=database, can_exist=can_exist
        )
        self.select = select

    @property
    def _prefix(self):
        return 'CREATE TABLE'

    @property
    def _pieces(self):
        yield 'AS ('
        yield self.select.compile()
        yield ')'


class CreateView(CTAS):

    """Create a view"""

    def __init__(self, table_name, select, database=None, can_exist=False):
        super(CreateView, self).__init__(
            table_name, select, database=database, can_exist=can_exist
        )

    @property
    def _pieces(self):
        yield 'AS'
        yield self.select.compile()

    @property
    def _prefix(self):
        return 'CREATE VIEW'


class LoadData(MapDDDL):

    """
    Generate DDL for LOAD DATA command. Cannot be cancelled
    """

    def __init__(
            self, table_name, path, database=None, overwrite=False):
        self.table_name = table_name
        self.database = database
        self.path = path
        self.overwrite = overwrite

    def compile(self):
        overwrite = 'OVERWRITE ' if self.overwrite else ''

        return (
            "LOAD DATA INPATH '{}' {}INTO TABLE {}"
            .format(self.path, overwrite, self.table_name)
        )


class AlterTable(MapDDDL):

    def __init__(self, table, tbl_properties=None):
        self.table = table
        self.tbl_properties = tbl_properties

    def _wrap_command(self, cmd):
        return 'ALTER TABLE {}'.format(cmd)

    def _format_properties(self, prefix=''):
        tokens = []

        if self.tbl_properties is not None:
            # tokens.append(format_tblproperties(self.tbl_properties))
            pass

        if len(tokens) > 0:
            return '\n{}{}'.format(prefix, '\n'.join(tokens))
        else:
            return ''

    def compile(self):
        props = self._format_properties()
        action = '{} SET {}'.format(self.table, props)
        return self._wrap_command(action)


class RenameTable(AlterTable):

    def __init__(self, old_name, new_name, old_database=None,
                 new_database=None):
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
        cmd = '{} RENAME TO {}'.format(self.old_qualified_name,
                                       self.new_qualified_name)
        return self._wrap_command(cmd)


class DropObject(MapDDDL):

    def __init__(self, must_exist=True):
        self.must_exist = must_exist

    def compile(self):
        if_exists = '' if self.must_exist else 'IF EXISTS '
        object_name = self._object_name()
        return 'DROP {} {}{}'.format(self._object_type, if_exists, object_name)


class DropTable(DropObject):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None, must_exist=True):
        super(DropTable, self).__init__(must_exist=must_exist)
        self.table_name = table_name
        self.database = database

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database)


class TruncateTable(MapDDDL):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self._get_scoped_name(self.table_name, self.database)
        return 'TRUNCATE TABLE {}'.format(name)


class DropView(DropTable):

    _object_type = 'VIEW'


class CacheTable(MapDDDL):

    def __init__(self, table_name, database=None, pool='default'):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return "ALTER TABLE {} SET CACHED IN '{}'" .format(
            scoped_name, self.pool
        )


class CreateDatabase(CreateDDL):

    def __init__(self, name, owner=None):
        self.name = name
        self.owner = owner

    def compile(self):
        name = quote_identifier(self.name)

        cmd = 'CREATE DATABASE'
        properties = ''

        if self.owner:
            properties = '(owner=\'{}\')'.format(self.owner)

        return '{} {} {}'.format(cmd, name, properties)


class DropDatabase(DropObject):

    _object_type = 'DATABASE'

    def __init__(self, name):
        super(DropDatabase, self).__init__(must_exist=True)
        self.name = name

    def _object_name(self):
        return self.name


def format_schema(schema):
    elements = [_format_schema_element(name, t)
                for name, t in zip(schema.names, schema.types)]
    return '({})'.format(',\n '.join(elements))


def _format_schema_element(name, t):
    return '{} {}'.format(
        quote_identifier(name, force=True), _type_to_sql_string(t)
    )


class InsertPandas(MapDDML):

    def __init__(self, table_name, df, insert_index=False, database=None):
        self.table_name = table_name
        self.database = database
        self.df = df.copy()

        if insert_index:
            self.df.reset_index(inplace=True)

    def _get_field_names(self):
        return self.df.keys()

    def compile(self):
        cmd = 'INSERT INTO'

        fields = self._get_field_names()
        scoped_name = self._get_scoped_name(self.table_name, self.database)

        return'{0} {1} ({2})\n{3}'.format(
            cmd, self.table_name
        )


def _mapd_input_signature(inputs):
    # TODO: varargs '{}...'.format(val)
    return ', '.join(map(_type_to_sql_string, inputs))
