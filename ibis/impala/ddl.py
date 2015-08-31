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

from ibis.compat import StringIO
import re

from ibis.sql.compiler import DDL
from .compiler import quote_identifier, _type_to_sql_string

from ibis.expr.datatypes import validate_type
from ibis.compat import py_string
import ibis.expr.rules as rules


fully_qualified_re = re.compile("(.*)\.(?:`(.*)`|(.*))")


def _is_fully_qualified(x):
    m = fully_qualified_re.search(x)
    return bool(m)


def _is_quoted(x):
    regex = re.compile("(?:`(.*)`|(.*))")
    quoted, unquoted = regex.match(x).groups()
    return quoted is not None


class ImpalaDDL(DDL):

    def _get_scoped_name(self, obj_name, database):
        if database:
            scoped_name = '{0}.`{1}`'.format(database, obj_name)
        else:
            if not _is_fully_qualified(obj_name):
                if _is_quoted(obj_name):
                    return obj_name
                else:
                    return '`{0}`'.format(obj_name)
            else:
                return obj_name
        return scoped_name


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
        buf = StringIO()
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
        buf = StringIO()
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
        buf = StringIO()
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

        buf = StringIO()
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
        buf = StringIO()

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

        buf = StringIO()
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
        buf = StringIO()
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


class AlterTable(ImpalaDDL):

    def _wrap_command(self, cmd):
        return 'ALTER TABLE {0}'.format(cmd)


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
        cmd = '{0} RENAME TO {1}'.format(self.old_qualified_name,
                                         self.new_qualified_name)
        return self._wrap_command(cmd)


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


class CreateFunctionBase(ImpalaDDL):

    _object_type = 'FUNCTION'

    def __init__(self, lib_path, inputs, output, name, database=None):
        self.lib_path = lib_path

        self.inputs, self.output = inputs, output
        self.input_sig = _impala_signature(inputs)
        self.output_sig = _arg_to_string(output)

        self.name = name
        self.database = database

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.name, self.database)
        return ('{0!s}({1!s}) returns {2!s}'
                .format(scoped_name, self.input_sig, self.output_sig))


class CreateFunction(CreateFunctionBase):

    def __init__(self, lib_path, so_symbol, inputs, output,
                 name, database=None):
        self.so_symbol = so_symbol

        CreateFunctionBase.__init__(self, lib_path, inputs, output,
                                    name, database=database)

    def compile(self):
        create_decl = 'CREATE FUNCTION'
        create_line = self._create_line()
        param_line = ("location '{0!s}' symbol='{1!s}'"
                      .format(self.lib_path, self.so_symbol))
        full_line = ' '.join([create_decl, create_line, param_line])
        return full_line


class CreateAggregateFunction(CreateFunction):

    def __init__(self, lib_path, inputs, output, update_fn, init_fn,
                 merge_fn, serialize_fn, finalize_fn, name, database):
        self.init = init_fn
        self.update = update_fn
        self.merge = merge_fn
        self.serialize = serialize_fn
        self.finalize = finalize_fn

        CreateFunctionBase.__init__(self, lib_path, inputs, output,
                                    name, database=database)

    def compile(self):
        create_decl = 'CREATE AGGREGATE FUNCTION'
        create_line = self._create_line()
        tokens = ["location '{0!s}'".format(self.lib_path)]

        if self.init is not None:
            tokens.append("init_fn='{0}'".format(self.init))

        tokens.append("update_fn='{0}'".format(self.update))

        if self.merge is not None:
            tokens.append("merge_fn='{0}'".format(self.merge))

        if self.serialize is not None:
            tokens.append("serialize_fn='{0}'".format(self.serialize))

        if self.finalize is not None:
            tokens.append("finalize_fn='{0}'".format(self.finalize))

        full_line = (' '.join([create_decl, create_line]) + ' ' +
                     '\n'.join(tokens))
        return full_line


class DropFunction(DropObject):

    def __init__(self, name, inputs, must_exist=True,
                 aggregate=False, database=None):
        self.name = name

        self.inputs = inputs
        self.input_sig = _impala_signature(inputs)

        self.must_exist = must_exist
        self.aggregate = aggregate
        self.database = database
        DropObject.__init__(self, must_exist=must_exist)

    def _object_name(self):
        return self.name

    def _function_sig(self):
        full_name = self._get_scoped_name(self.name, self.database)
        return '{0!s}({1!s})'.format(full_name, self.input_sig)

    def compile(self):
        tokens = ['DROP']
        if self.aggregate:
            tokens.append('AGGREGATE')
        tokens.append('FUNCTION')
        if not self.must_exist:
            tokens.append('IF EXISTS')

        tokens.append(self._function_sig())
        return ' '.join(tokens)


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


def _impala_signature(sig):
    if isinstance(sig, rules.TypeSignature):
        if isinstance(sig, rules.VarArgs):
            val = _arg_to_string(sig.arg_type)
            return '{0}...'.format(val)
        else:
            return ', '.join([_arg_to_string(arg) for arg in sig.types])
    else:
        return ', '.join([_type_to_sql_string(validate_type(x))
                          for x in sig])


def _arg_to_string(arg):
    if isinstance(arg, rules.ValueTyped):
        types = arg.types
        if len(types) > 1:
            raise NotImplementedError
        return _type_to_sql_string(types[0])
    elif isinstance(arg, py_string):
        return _type_to_sql_string(validate_type(arg))
    else:
        raise NotImplementedError
