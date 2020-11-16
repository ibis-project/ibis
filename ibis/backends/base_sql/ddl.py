import re

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base_sql import quote_identifier, type_to_sql_string
from ibis.backends.base_sqlalchemy.compiler import DDL, DML

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")
_format_aliases = {'TEXT': 'TEXTFILE'}


def _sanitize_format(format):
    if format is None:
        return
    format = format.upper()
    format = _format_aliases.get(format, format)
    if format not in ('PARQUET', 'AVRO', 'TEXTFILE'):
        raise ValueError('Invalid format: {!r}'.format(format))

    return format


def is_fully_qualified(x):
    return bool(fully_qualified_re.search(x))


def _is_quoted(x):
    regex = re.compile(r"(?:`(.*)`|(.*))")
    quoted, _ = regex.match(x).groups()
    return quoted is not None


def format_schema(schema):
    elements = [
        _format_schema_element(name, t)
        for name, t in zip(schema.names, schema.types)
    ]
    return '({})'.format(',\n '.join(elements))


def _format_schema_element(name, t):
    return '{} {}'.format(
        quote_identifier(name, force=True), type_to_sql_string(t),
    )


def _format_partition_kv(k, v, type):
    if type == dt.string:
        value_formatted = '"{}"'.format(v)
    else:
        value_formatted = str(v)

    return '{}={}'.format(k, value_formatted)


def format_partition(partition, partition_schema):
    tokens = []
    if isinstance(partition, dict):
        for name in partition_schema:
            if name in partition:
                tok = _format_partition_kv(
                    name, partition[name], partition_schema[name]
                )
            else:
                # dynamic partitioning
                tok = name
            tokens.append(tok)
    else:
        for name, value in zip(partition_schema, partition):
            tok = _format_partition_kv(name, value, partition_schema[name])
            tokens.append(tok)

    return 'PARTITION ({})'.format(', '.join(tokens))


def format_properties(props):
    tokens = []
    for k, v in sorted(props.items()):
        tokens.append("  '{}'='{}'".format(k, v))

    return '(\n{}\n)'.format(',\n'.join(tokens))


def format_tblproperties(props):
    formatted_props = format_properties(props)
    return 'TBLPROPERTIES {}'.format(formatted_props)


def _serdeproperties(props):
    formatted_props = format_properties(props)
    return 'SERDEPROPERTIES {}'.format(formatted_props)


class BaseQualifiedSQLStatement:
    def _get_scoped_name(self, obj_name, database):
        if database:
            scoped_name = '{}.`{}`'.format(database, obj_name)
        else:
            if not is_fully_qualified(obj_name):
                if _is_quoted(obj_name):
                    return obj_name
                else:
                    return '`{}`'.format(obj_name)
            else:
                return obj_name
        return scoped_name


class BaseDDL(DDL, BaseQualifiedSQLStatement):
    pass


class BaseDML(DML, BaseQualifiedSQLStatement):
    pass


class CreateDDL(BaseDDL):
    def _if_exists(self):
        return 'IF NOT EXISTS ' if self.can_exist else ''


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
        external=False,
        format='parquet',
        can_exist=False,
        partition=None,
        path=None,
        tbl_properties=None,
    ):
        self.table_name = table_name
        self.database = database
        self.partition = partition
        self.path = path
        self.external = external
        self.can_exist = can_exist
        self.format = _sanitize_format(format)
        self.tbl_properties = tbl_properties

    @property
    def _prefix(self):
        if self.external:
            return 'CREATE EXTERNAL TABLE'
        else:
            return 'CREATE TABLE'

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return '{} {}{}'.format(self._prefix, self._if_exists(), scoped_name)

    def _location(self):
        return "LOCATION '{}'".format(self.path) if self.path else None

    def _storage(self):
        # By the time we're here, we have a valid format
        return 'STORED AS {}'.format(self.format)

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
        self,
        table_name,
        select,
        database=None,
        external=False,
        format='parquet',
        can_exist=False,
        path=None,
        partition=None,
    ):
        super().__init__(
            table_name,
            database=database,
            external=external,
            format=format,
            can_exist=can_exist,
            path=path,
            partition=partition,
        )
        self.select = select

    @property
    def _pieces(self):
        yield self._partitioned_by()
        yield self._storage()
        yield self._location()
        yield 'AS'
        yield self.select.compile()

    def _partitioned_by(self):
        if self.partition is not None:
            return 'PARTITIONED BY ({})'.format(
                ', '.join(
                    quote_identifier(expr._name) for expr in self.partition
                )
            )
        return None


class CreateView(CTAS):

    """Create a view"""

    def __init__(self, table_name, select, database=None, can_exist=False):
        super().__init__(
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
        super().__init__(table_name, **kwargs)
        self.schema = schema
        self.table_format = table_format

    @property
    def _pieces(self):
        if self.partition is not None:
            main_schema = self.schema
            part_schema = self.partition
            if not isinstance(part_schema, sch.Schema):
                part_schema = sch.Schema(
                    part_schema, [self.schema[name] for name in part_schema]
                )

            to_delete = []
            for name in self.partition:
                if name in self.schema:
                    to_delete.append(name)

            if len(to_delete):
                main_schema = main_schema.delete(to_delete)

            yield format_schema(main_schema)
            yield 'PARTITIONED BY {}'.format(format_schema(part_schema))
        else:
            yield format_schema(self.schema)

        if self.table_format is not None:
            yield '\n'.join(self.table_format.to_ddl())
        else:
            yield self._storage()

        yield self._location()


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


class DropObject(BaseDDL):
    def __init__(self, must_exist=True):
        self.must_exist = must_exist

    def compile(self):
        if_exists = '' if self.must_exist else 'IF EXISTS '
        object_name = self._object_name()
        return 'DROP {} {}{}'.format(self._object_type, if_exists, object_name)


class DropDatabase(DropObject):

    _object_type = 'DATABASE'

    def __init__(self, name, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.name = name

    def _object_name(self):
        return self.name


class DropTable(DropObject):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.table_name = table_name
        self.database = database

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database)


class DropView(DropTable):

    _object_type = 'VIEW'


class TruncateTable(BaseDDL):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self._get_scoped_name(self.table_name, self.database)
        return 'TRUNCATE TABLE {}'.format(name)


class InsertSelect(BaseDML):
    def __init__(
        self,
        table_name,
        select_expr,
        database=None,
        partition=None,
        partition_schema=None,
        overwrite=False,
    ):
        self.table_name = table_name
        self.database = database
        self.select = select_expr

        self.partition = partition
        self.partition_schema = partition_schema

        self.overwrite = overwrite

    def compile(self):
        if self.overwrite:
            cmd = 'INSERT OVERWRITE'
        else:
            cmd = 'INSERT INTO'

        if self.partition is not None:
            part = format_partition(self.partition, self.partition_schema)
            partition = ' {} '.format(part)
        else:
            partition = ''

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return '{0} {1}{2}\n{3}'.format(
            cmd, scoped_name, partition, select_query
        )


class AlterTable(BaseDDL):
    def __init__(
        self,
        table,
        location=None,
        format=None,
        tbl_properties=None,
        serde_properties=None,
    ):
        self.table = table
        self.location = location
        self.format = _sanitize_format(format)
        self.tbl_properties = tbl_properties
        self.serde_properties = serde_properties

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

        if self.serde_properties is not None:
            tokens.append(_serdeproperties(self.serde_properties))

        if len(tokens) > 0:
            return '\n{}{}'.format(prefix, '\n'.join(tokens))
        else:
            return ''

    def compile(self):
        props = self._format_properties()
        action = '{} SET {}'.format(self.table, props)
        return self._wrap_command(action)


class DropFunction(DropObject):
    def __init__(
        self, name, inputs, must_exist=True, aggregate=False, database=None
    ):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.inputs = tuple(map(dt.dtype, inputs))
        self.must_exist = must_exist
        self.aggregate = aggregate
        self.database = database

    def _object_name(self):
        return self.name

    def compile(self):
        tokens = ['DROP']
        if self.aggregate:
            tokens.append('AGGREGATE')
        tokens.append('FUNCTION')
        if not self.must_exist:
            tokens.append('IF EXISTS')

        tokens.append(self._impala_signature())
        return ' '.join(tokens)


class RenameTable(AlterTable):
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
        cmd = '{} RENAME TO {}'.format(
            self.old_qualified_name, self.new_qualified_name
        )
        return self._wrap_command(cmd)
