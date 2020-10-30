from ibis.backends.base_sql import ddl as base_ddl
from ibis.backends.base_sql import quote_identifier

from .compiler import _type_to_sql_string

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


class CreateTable(base_ddl.CreateTable):

    """Create a table"""

    def __init__(
        self,
        table_name,
        database=None,
        format='parquet',
        can_exist=False,
        tbl_properties=None,
    ):
        super().__init__(
            table_name,
            database=database,
            external=False,
            format=format,
            can_exist=can_exist,
            partition=None,
            tbl_properties=tbl_properties,
        )

    def _storage(self):
        return 'USING {}'.format(self.format)


class CreateTableWithSchema(base_ddl.CreateTableWithSchema):
    def _storage(self):
        return 'USING {}'.format(self.format)


class CTAS(base_ddl.CTAS):

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
    ):
        super().__init__(
            table_name,
            select,
            database=database,
            format=format,
            can_exist=can_exist,
        )
        self.select = select

    def _storage(self):
        return 'USING {}'.format(self.format)


class CreateView(CTAS):

    """Create a view"""

    def __init__(
        self,
        table_name,
        select,
        database=None,
        can_exist=False,
        temporary=False,
    ):
        super().__init__(
            table_name, select, database=database, can_exist=can_exist
        )
        self.temporary = temporary

    @property
    def _pieces(self):
        yield 'AS'
        yield self.select.compile()

    @property
    def _prefix(self):
        return 'CREATE {}{}VIEW'.format(
            self._or_replace_clause(), self._temporary_clause()
        )

    def _or_replace_clause(self):
        return 'OR REPLACE ' if self.can_exist else ''

    def _temporary_clause(self):
        return 'TEMPORARY ' if self.temporary else ''

    def _if_exists(self):
        return ''


def format_schema(schema):
    elements = [
        _format_schema_element(name, t)
        for name, t in zip(schema.names, schema.types)
    ]
    return '({})'.format(',\n '.join(elements))


def _format_schema_element(name, t):
    return '{} {}'.format(
        quote_identifier(name, force=True), _type_to_sql_string(t),
    )


class DropDatabase(base_ddl.DropObject):

    _object_type = 'DATABASE'

    def __init__(self, name, must_exist=True, cascade=False):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.cascade = cascade

    def _object_name(self):
        return self.name

    def compile(self):
        compiled = super().compile()
        if self.cascade:
            return '{} CASCADE'.format(compiled)
        else:
            return compiled


class DropFunction(base_ddl.DropObject):

    _object_type = 'TEMPORARY FUNCTION'

    def __init__(self, name, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.must_exist = must_exist

    def _object_name(self):
        return self.name


class InsertSelect(base_ddl.InsertSelect):
    def __init__(
        self, table_name, select_expr, database=None, overwrite=False
    ):
        super().__init__(
            table_name,
            select_expr,
            database=database,
            partition=None,
            partition_schema=None,
            overwrite=overwrite,
        )

    def compile(self):
        if self.overwrite:
            cmd = 'INSERT OVERWRITE TABLE'
        else:
            cmd = 'INSERT INTO'

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return '{0} {1}\n{2}'.format(cmd, scoped_name, select_query)


class AlterTable(base_ddl.AlterTable):
    def __init__(self, table, tbl_properties=None):
        super().__init__(
            table,
            location=None,
            format=None,
            tbl_properties=tbl_properties,
            serde_properties=None,
        )

    def compile(self):
        props = self._format_properties()
        action = '{} SET{}'.format(self.table, props)
        return self._wrap_command(action)


class RenameTable(base_ddl.RenameTable):
    def __init__(self, old_name, new_name):
        super().__init__(
            old_name, new_name, old_database=None, new_database=None
        )
