import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops

# known SQL keywords that need to be quoted to use as identifiers
base_identifiers = [
    'add',
    'aggregate',
    'all',
    'alter',
    'and',
    'api_version',
    'as',
    'asc',
    'avro',
    'between',
    'bigint',
    'binary',
    'boolean',
    'by',
    'cached',
    'case',
    'cast',
    'change',
    'char',
    'class',
    'close_fn',
    'column',
    'columns',
    'comment',
    'compute',
    'create',
    'cross',
    'data',
    'database',
    'databases',
    'date',
    'datetime',
    'decimal',
    'delimited',
    'desc',
    'describe',
    'distinct',
    'div',
    'double',
    'drop',
    'else',
    'end',
    'escaped',
    'exists',
    'explain',
    'external',
    'false',
    'fields',
    'fileformat',
    'finalize_fn',
    'first',
    'float',
    'format',
    'formatted',
    'from',
    'full',
    'function',
    'functions',
    'group',
    'having',
    'if',
    'in',
    'incremental',
    'init_fn',
    'inner',
    'inpath',
    'insert',
    'int',
    'integer',
    'intermediate',
    'interval',
    'into',
    'invalidate',
    'is',
    'join',
    'last',
    'left',
    'like',
    'limit',
    'lines',
    'load',
    'location',
    'merge_fn',
    'metadata',
    'not',
    'null',
    'nulls',
    'offset',
    'on',
    'or',
    'order',
    'outer',
    'overwrite',
    'parquet',
    'parquetfile',
    'partition',
    'partitioned',
    'partitions',
    'prepare_fn',
    'produced',
    'rcfile',
    'real',
    'refresh',
    'regexp',
    'rename',
    'replace',
    'returns',
    'right',
    'rlike',
    'row',
    'schema',
    'schemas',
    'select',
    'semi',
    'sequencefile',
    'serdeproperties',
    'serialize_fn',
    'set',
    'show',
    'smallint',
    'stats',
    'stored',
    'straight_join',
    'string',
    'symbol',
    'table',
    'tables',
    'tblproperties',
    'terminated',
    'textfile',
    'then',
    'timestamp',
    'tinyint',
    'to',
    'true',
    'uncached',
    'union',
    'update_fn',
    'use',
    'using',
    'values',
    'view',
    'when',
    'where',
    'with',
]


def register():
    # decorator to register operations
    pass


class BaseOperationRegistry:
    backend_kind = None  # One of 'sql', 'sqlalchemy' # TODO maybe others?
    # Used to quote identifiers (e.g. names of columns with spaces')
    quote = '`'
    _translate_functions = {}
    context_class = QueryContext

    @classmethod
    def _translate_function(cls, operation):
        """
        Find the translate function for an operation.

        For legacy reasons, two different ways of specifying translate
        functions are supported:

        The first one tried is a method of this class:

        >>> class MyBackendOperationRegistry(BaseOperationRegistry):
        ...     @register(ops.HelloWorld)
        ...     def hello_world(cls, expr):
        ...         return 'hello world'

        The legacy one is an entry in the `_translate_functions` attribute:

        >>> MyBackendOperationRegistry._translate_functions[ops.Hello_World]
        """
        try:
            translate_function = cls._translate_functions[operation]
        except KeyError:
            raise com.OperationNotDefinedError(f'No translation rule for {op}')

    @classmethod
    def _set_name(cls, translated_expression, name):
        if cls.backend_kind == 'sql':
            if name.count(' ') or name.lower() in base_identifiers:
                name = f'`{name}`'
            return f'{translated_expression} AS {name}'
        elif cls.backend_kind == 'sqlalchemy':
            if hasattr(translated_expression, 'label'):
                return translated_expression.label(name)
        else:
            return translated_expression

    @classmethod
    def translate(cls, expr):
        name = expr.get_name()

        # A translation function can return another expression. This was before
        # handled with two sets of translation functions, `registry` and
        # `rewrites` but now we simply keep processing the return until it's
        # not an expression anymore
        while isinstance(expr, Expression):
            op = type(expr.op())
            try:
                translate_function = cls._translate_functions[op]
            except KeyError:
                raise com.OperationNotDefinedError(
                    f'No translation rule for {op}'
                )

            expr = translate_function(cls, expr)

        return cls._set_name(translated_expression=expr, name=name)

    # operations

    @register(ops.TableNode)
    def table_node(cls, expr):
        return '*'

    @register(ops.ScalarParameter)
    def scalar_parameter(cls, expr):
        raw_value = cls.context.params[expr.op()]
        return ibis.literal(raw_value, type=expr.type())
