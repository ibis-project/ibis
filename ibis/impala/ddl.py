# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import BytesIO
import re

from ibis.sql.ddl import DDL, Select
from ibis.sql.exprs import quote_identifier, _type_to_sql_string


fully_qualified_re = re.compile("(.*)\.(?:`(.*)`|(.*))")


def _is_fully_qualified(x):
    m = fully_qualified_re.search(x)
    return bool(m)


def _is_quoted(x):
    regex = re.compile("(?:`(.*)`|(.*))")
    quoted, unquoted = regex.match(x).groups()
    return quoted is not None


class ImpalaDDL(DDL):

    def _get_scoped_name(self, table_name, database):
        if database:
            scoped_name = '{0}.`{1}`'.format(database, table_name)
        else:
            if not _is_fully_qualified(table_name):
                if _is_quoted(table_name):
                    return table_name
                else:
                    return '`{0}`'.format(table_name)
            else:
                return table_name
        return scoped_name


class ImpalaSelect(Select):
    pass


class CreateDDL(ImpalaDDL):

    def _if_exists(self):
        return 'IF NOT EXISTS ' if self.can_exist else ''


class CreateTable(CreateDDL):

    """

    Parameters
    ----------
    partition :

    """

    def __init__(self, table_name, database=None, external=False,
                 format='parquet', can_exist=False,
                 partition=None, path=None):
        self.table_name = table_name
        self.database = database
        self.partition = partition
        self.path = path
        self.external = external
        self.can_exist = can_exist
        self.format = self._validate_storage_format(format)

    def _validate_storage_format(self, format):
        format = format.lower()
        if format not in ('parquet', 'avro'):
            raise ValueError('Invalid format: {0}'.format(format))
        return format

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)

        if self.external:
            create_decl = 'CREATE EXTERNAL TABLE'
        else:
            create_decl = 'CREATE TABLE'

        create_line = '{0} {1}{2}'.format(create_decl, self._if_exists(),
                                          scoped_name)
        return create_line

    def _location(self):
        if self.path:
            return "\nLOCATION '{0}'".format(self.path)
        return ''

    def _storage(self):
        storage_lines = {
            'parquet': '\nSTORED AS PARQUET',
            'avro': '\nSTORED AS AVRO'
        }
        return storage_lines[self.format]


class CTAS(CreateTable):

    """
    Create Table As Select
    """

    def __init__(self, table_name, select, database=None,
                 external=False, format='parquet', can_exist=False,
                 path=None):
        self.select = select
        CreateTable.__init__(self, table_name, database=database,
                             external=external, format=format,
                             can_exist=can_exist, path=path)

    def compile(self):
        buf = BytesIO()
        buf.write(self._create_line())
        buf.write(self._storage())
        buf.write(self._location())

        select_query = self.select.compile()
        buf.write('\nAS\n{0}'.format(select_query))
        return buf.getvalue()


class CreateView(CreateDDL):

    """
    Create Table As Select
    """

    def __init__(self, name, select, database=None, can_exist=False):
        self.name = name
        self.database = database
        self.select = select
        self.can_exist = can_exist

    def compile(self):
        buf = BytesIO()
        buf.write(self._create_line())

        select_query = self.select.compile()
        buf.write('\nAS\n{0}'.format(select_query))
        return buf.getvalue()

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.name, self.database)
        return '{0} {1}{2}'.format('CREATE VIEW', self._if_exists(),
                                   scoped_name)


class CreateTableParquet(CreateTable):

    def __init__(self, table_name, path,
                 example_file=None,
                 example_table=None,
                 schema=None,
                 external=True,
                 **kwargs):
        self.example_file = example_file
        self.example_table = example_table
        self.schema = schema
        CreateTable.__init__(self, table_name, external=external,
                             format='parquet', path=path, **kwargs)

        self._validate()

    def _validate(self):
        pass

    def compile(self):
        buf = BytesIO()
        buf.write(self._create_line())

        if self.example_file is not None:
            buf.write("\nLIKE PARQUET '{0}'".format(self.example_file))
        elif self.example_table is not None:
            buf.write("\nLIKE {0}".format(self.example_table))
        elif self.schema is not None:
            schema = format_schema(self.schema)
            buf.write('\n{0}'.format(schema))
        else:
            raise NotImplementedError

        buf.write(self._storage())
        buf.write(self._location())
        return buf.getvalue()


class CreateTableWithSchema(CreateTable):

    def __init__(self, table_name, schema, table_format, **kwargs):
        self.schema = schema
        self.table_format = table_format

        CreateTable.__init__(self, table_name, **kwargs)

    def compile(self):
        from ibis.expr.api import schema

        buf = BytesIO()
        buf.write(self._create_line())

        def _push_schema(x):
            formatted = format_schema(x)
            buf.write('{0}'.format(formatted))

        if self.partition is not None:
            modified_schema = []
            partition_schema = []
            for name, dtype in zip(self.schema.names, self.schema.types):
                if name in self.partition:
                    partition_schema.append((name, dtype))
                else:
                    modified_schema.append((name, dtype))

            buf.write('\n')
            _push_schema(schema(modified_schema))
            buf.write('\nPARTITIONED BY ')
            _push_schema(schema(partition_schema))
        else:
            buf.write('\n')
            _push_schema(self.schema)

        format_ddl = self.table_format.to_ddl()
        if format_ddl:
            buf.write(format_ddl)

        buf.write(self._location())

        return buf.getvalue()


class NoFormat(object):

    def to_ddl(self):
        return None


class DelimitedFormat(object):

    def __init__(self, path, delimiter=None, escapechar=None,
                 lineterminator=None):
        self.path = path
        self.delimiter = delimiter
        self.escapechar = escapechar
        self.lineterminator = lineterminator

    def to_ddl(self):
        buf = BytesIO()

        buf.write("\nROW FORMAT DELIMITED")

        if self.delimiter is not None:
            buf.write("\nFIELDS TERMINATED BY '{0}'".format(self.delimiter))

        if self.escapechar is not None:
            buf.write("\nESCAPED BY '{0}'".format(self.escapechar))

        if self.lineterminator is not None:
            buf.write("\nLINES TERMINATED BY '{0}'"
                      .format(self.lineterminator))

        buf.write("\nLOCATION '{0}'".format(self.path))

        return buf.getvalue()


class AvroFormat(object):

    def __init__(self, path, avro_schema):
        self.path = path
        self.avro_schema = avro_schema

    def to_ddl(self):
        import json

        buf = BytesIO()
        buf.write('\nSTORED AS AVRO')
        buf.write("\nLOCATION '{0}'".format(self.path))

        schema = json.dumps(self.avro_schema, indent=2, sort_keys=True)
        schema = '\n'.join([x.rstrip() for x in schema.split('\n')])
        buf.write("\nTBLPROPERTIES ('avro.schema.literal'='{0}')"
                  .format(schema))

        return buf.getvalue()


class CreateTableDelimited(CreateTableWithSchema):

    def __init__(self, table_name, path, schema,
                 delimiter=None, escapechar=None, lineterminator=None,
                 external=True, **kwargs):
        table_format = DelimitedFormat(path, delimiter=delimiter,
                                       escapechar=escapechar,
                                       lineterminator=lineterminator)
        CreateTableWithSchema.__init__(self, table_name, schema,
                                       table_format, external=external,
                                       **kwargs)


class CreateTableAvro(CreateTable):

    def __init__(self, table_name, path, avro_schema, external=True, **kwargs):
        self.table_format = AvroFormat(path, avro_schema)

        CreateTable.__init__(self, table_name, external=external, **kwargs)

    def compile(self):
        buf = BytesIO()
        buf.write(self._create_line())

        format_ddl = self.table_format.to_ddl()
        buf.write(format_ddl)

        return buf.getvalue()


class InsertSelect(ImpalaDDL):

    def __init__(self, table_name, select_expr, database=None,
                 overwrite=False):
        self.table_name = table_name
        self.database = database
        self.select = select_expr

        self.overwrite = overwrite

    def compile(self):
        if self.overwrite:
            cmd = 'INSERT OVERWRITE'
        else:
            cmd = 'INSERT INTO'

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return'{0} {1}\n{2}'.format(cmd, scoped_name, select_query)


class DropObject(ImpalaDDL):

    def __init__(self, must_exist=True):
        self.must_exist = must_exist

    def compile(self):
        if_exists = '' if self.must_exist else 'IF EXISTS '
        object_name = self._object_name()
        drop_line = 'DROP {0} {1}{2}'.format(self._object_type, if_exists,
                                             object_name)
        return drop_line


class DropTable(DropObject):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None, must_exist=True):
        self.table_name = table_name
        self.database = database
        DropObject.__init__(self, must_exist=must_exist)

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database)


class TruncateTable(ImpalaDDL):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self._get_scoped_name(self.table_name, self.database)
        return 'TRUNCATE TABLE {0}'.format(name)


class DropView(DropTable):

    _object_type = 'VIEW'


class CacheTable(ImpalaDDL):

    def __init__(self, table_name, database=None, pool='default'):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        cache_line = ('ALTER TABLE {0} SET CACHED IN \'{1}\''
                      .format(scoped_name, self.pool))
        return cache_line


class CreateDatabase(CreateDDL):

    def __init__(self, name, path=None, can_exist=False):
        self.name = name
        self.path = path
        self.can_exist = can_exist

    def compile(self):
        name = quote_identifier(self.name)

        create_decl = 'CREATE DATABASE'
        create_line = '{0} {1}{2}'.format(create_decl, self._if_exists(),
                                          name)
        if self.path is not None:
            create_line += "\nLOCATION '{0}'".format(self.path)

        return create_line


class DropDatabase(DropObject):

    _object_type = 'DATABASE'

    def __init__(self, name, must_exist=True):
        self.name = name
        DropObject.__init__(self, must_exist=must_exist)

    def _object_name(self):
        return self.name


def format_schema(schema):
    elements = [_format_schema_element(name, t)
                for name, t in zip(schema.names, schema.types)]
    return '({0})'.format(',\n '.join(elements))


def _format_schema_element(name, t):
    return '{0} {1}'.format(quote_identifier(name, force=True),
                            _type_to_sql_string(t))


class CreateFunction(ImpalaDDL):

    _object_type = 'FUNCTION'

    def __init__(self, hdfs_file, so_symbol, inputs, output,
                 name, database=None):
        self.hdfs_file = hdfs_file
        self.so_symbol = so_symbol
        self.inputs = _impala_signature(inputs)
        self.output = _impala_signature([output])[0]
        self.name = name
        self.database = database

    def _get_scoped_name(self):
        if self.database:
            return '{0}.{1}'.format(self.database, self.name)
        else:
            return self.name

    def compile(self):
        create_decl = 'CREATE FUNCTION'
        scoped_name = self._get_scoped_name()
        create_line = ('{0!s}({1!s}) returns {2!s}'
                       .format(scoped_name, ', '.join(self.inputs),
                               self.output))
        param_line = "location '{0!s}' symbol='{1!s}'".format(self.hdfs_file,
                                                              self.so_symbol)
        full_line = ' '.join([create_decl, create_line, param_line])
        return full_line


class CreateAggregateFunction(ImpalaDDL):

    _object_type = 'FUNCTION'

    def __init__(self, hdfs_file, inputs, output, init_fn, update_fn,
                 merge_fn, serialize_fn, finalize_fn, name, database):
        self.hdfs_file = hdfs_file
        self.inputs = _impala_signature(inputs)
        self.output = _impala_signature([output])[0]
        self.init = init_fn
        self.update = update_fn
        self.merge = merge_fn
        self.serialize = serialize_fn
        self.finalize = finalize_fn

        self.name = name
        self.database = database

    def _get_scoped_name(self):
        if self.database:
            return '{0}.{1}'.format(self.database, self.name)
        else:
            return self.name

    def compile(self):
        create_decl = 'CREATE AGGREGATE FUNCTION'
        scoped_name = self._get_scoped_name()
        create_line = ('{0!s}({1!s}) returns {2!s}'
                       .format(scoped_name, ', '.join(self.inputs),
                               self.output))
        tokens = ["location '{0!s}'".format(self.hdfs_file),
                  "init_fn='{0}'".format(self.init),
                  "update_fn='{0}'".format(self.update),
                  "merge_fn='{0}'".format(self.merge),
                  "finalize_fn='{0}'".format(self.finalize)]

        if self.serialize is not None:
            tokens.append("serialize_fn='{0}'".format(self.serialize))

        full_line = (' '.join([create_decl, create_line]) + ' ' +
                     '\n'.join(tokens))
        return full_line


class DropFunction(DropObject):

    def __init__(self, name, input_types, must_exist=True,
                 aggregate=False, database=None):
        self.name = name
        self.inputs = _impala_signature(input_types)
        self.must_exist = must_exist
        self.aggregate = aggregate
        self.database = database
        DropObject.__init__(self, must_exist=must_exist)

    def _object_name(self):
        return self.name

    def _get_scoped_name(self):
        if self.database:
            return '{0}.{1}'.format(self.database, self.name)
        else:
            return self.name

    def compile(self):
        statement = 'DROP'
        if self.aggregate:
            statement += ' AGGREGATE'
        statement += ' FUNCTION'
        if not self.must_exist:
            statement += ' IF EXISTS'
        full_name = self._get_scoped_name()
        func_line = ' {0!s}({1!s})'.format(full_name, ', '.join(self.inputs))
        statement += func_line
        return statement


class ListFunction(ImpalaDDL):

    def __init__(self, database, like=None, aggregate=False):
        self.database = database
        self.like = like
        self.aggregate = aggregate

    def compile(self):
        statement = 'SHOW '
        if self.aggregate:
            statement += 'AGGREGATE '
        statement += 'FUNCTIONS IN {0}'.format(self.database)
        if self.like:
            statement += " LIKE '{0}'".format(self.like)
        return statement


def _impala_signature(types):
    from ibis.expr.datatypes import validate_type
    return [_type_to_sql_string(validate_type(x)) for x in types]
