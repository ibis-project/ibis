from ibis.sql.compiler import DDL, DML
from .compiler import quote_identifier, _type_to_sql_string

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
    """Create DDL"""


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


def _format_properties(props):
    tokens = []
    for k, v in sorted(props.items()):
        tokens.append("  '{}'='{}'".format(k, v))

    return '(\n{}\n)'.format(',\n'.join(tokens))


class CreateTable(CreateDDL):
    """

    Parameters
    ----------
    table_name : str
    database : str
    """

    def __init__(
        self, table_name, database=None
    ):
        self.table_name = table_name
        self.database = database

    @property
    def _prefix(self):
        return 'CREATE TABLE'

    def _create_line(self):
        return '{} {}'.format(
            self._prefix, self.table_name
        )

    @property
    def pieces(self):
        yield self._create_line()
        for piece in filter(None, self._pieces):
            yield piece

    def compile(self):
        return '\n'.join(self.pieces)


class CreateTableWithSchema(CreateTable):
    def __init__(
        self, table_name, schema, database=None, fragment_size=None,
        max_rows=None, page_size=None, partitions=None, shard_count=None
    ):
        self.table_name = table_name
        self.database = database
        self.schema = schema
        self.fragment_size = fragment_size
        self.max_rows = max_rows
        self.page_size = page_size
        self.partitions = partitions
        self.shard_count = shard_count

    @property
    def with_params(self):
        return dict(
            fragment_size=self.fragment_size,
            max_rows=self.max_rows,
            page_size=self.page_size,
            partitions=self.partitions,
            shard_count=self.shard_count
        )

    @property
    def _pieces(self):
        yield format_schema(self.schema)

        with_stmt = ','.join([
            '{}={}'.format(i, "'{}'".format(v) if isinstance(v, str) else v)
            for i, v in self.with_params.items() if v is not None
        ])

        if with_stmt:
            yield ' WITH ({})'.format(with_stmt)


class CTAS(CreateTable):
    """
    Create Table As Select
    """

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
    """Create a view"""

    def __init__(self, table_name, select, database=None):
        super(CreateView, self).__init__(table_name, select, database=database)

    @property
    def _pieces(self):
        yield 'AS'
        yield self.select.compile()

    @property
    def _prefix(self):
        return 'CREATE VIEW'


class DropView(DropTable):
    _object_type = 'VIEW'


# USER

class AlterUser(MapDDDL):
    """Create user"""

    def __init__(
        self, name, password=None, database=None, is_super=False,
        insert_access=None
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
        yield 'ALTER USER {} ('.format(self.name)
        yield ','.join(self._params)
        yield ')'

    def compile(self):
        return '\n'.join(self.pieces)


class CreateUser(MapDDDL):
    """Create user"""

    def __init__(self, name, password, database=None, is_super=False):
        self.name = name
        self.password = password
        self.database = database
        self.is_super = is_super

    @property
    def pieces(self):
        yield 'CREATE USER {} ('.format(self.name)
        yield "  password='{}',".format(self.password)
        yield "  is_super='{}'".format('true' if self.is_super else 'false')
        yield ')'

    def compile(self):
        return '\n'.join(self.pieces)


class DropUser(MapDDDL):
    """Create user"""

    def __init__(self, name, database=None):
        self.name = name
        self.database = database

    @property
    def pieces(self):
        yield 'DROP USER {}'.format(self.name)

    def compile(self):
        return '\n'.join(self.pieces)


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


class TruncateTable(MapDDDL):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self._get_scoped_name(self.table_name, self.database)
        return 'TRUNCATE TABLE {}'.format(name)


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
    elements = [
        _format_schema_element(name, t)
        for name, t in zip(schema.names, schema.types)
    ]
    return '({})'.format(',\n '.join(elements))


def _format_schema_element(name, t):
    return '{} {}'.format(
        quote_identifier(name, force=False), _type_to_sql_string(t)
    )


class InsertPandas(MapDDML):

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
        cmd = 'INSERT INTO'

        fields = self._get_field_names()

        stmt = '{0} {1} ({2}) VALUES '.format(
            cmd, self.table_name, fields
        )

        for values in self._get_field_values():
            yield '{} ({});'.format(stmt, ','.join(values))

    def compile(self):
        return '\n'.join(self.pieces)


def _mapd_input_signature(inputs):
    # TODO: varargs '{}...'.format(val)
    return ', '.join(map(_type_to_sql_string, inputs))
