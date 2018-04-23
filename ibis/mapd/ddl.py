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


class CreateTableWithSchema(CreateTable):

    def __init__(self, table_name, schema, table_format=None, **kwargs):
        super(CreateTableWithSchema, self).__init__(table_name, **kwargs)
        self.schema = schema
        self.table_format = table_format

    @property
    def _pieces(self):
        if self.partition is not None:
            main_schema = self.schema
            part_schema = self.partition
            if not isinstance(part_schema, sch.Schema):
                part_schema = sch.Schema(
                    part_schema,
                    [self.schema[name] for name in part_schema])

            to_delete = []
            for name in self.partition:
                if name in self.schema:
                    to_delete.append(name)

            if len(to_delete):
                main_schema = main_schema.delete(to_delete)

            yield format_schema(main_schema)
        else:
            yield format_schema(self.schema)

        if self.table_format is not None:
            yield '\n'.join(self.table_format.to_ddl())
        else:
            yield self._storage()

        yield self._location()


class InsertSelect(MapDDML):

    def __init__(
        self, table_name, select_expr, database=None
    ):
        self.table_name = table_name
        self.database = database
        self.select = select_expr

    def compile(self):
        cmd = 'INSERT INTO'

        select_query = self.select.compile()
        return'{0} {1}\n{2}'.format(
            cmd, self.table_name, select_query
        )


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


class DropDatabase(DropObject):

    _object_type = 'DATABASE'

    def __init__(self, name, must_exist=True):
        super(DropDatabase, self).__init__(must_exist=must_exist)
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


class CreateFunction(MapDDDL):

    _object_type = 'FUNCTION'

    def __init__(self, func, name=None, database=None):
        self.func = func
        self.name = name or func.name
        self.database = database

    def _mapd_signature(self):
        scoped_name = self._get_scoped_name(self.name, self.database)
        input_sig = _mapd_input_signature(self.func.inputs)
        output_sig = _type_to_sql_string(self.func.output)

        return '{}({}) returns {}'.format(scoped_name, input_sig, output_sig)


def _mapd_input_signature(inputs):
    # TODO: varargs '{}...'.format(val)
    return ', '.join(map(_type_to_sql_string, inputs))
