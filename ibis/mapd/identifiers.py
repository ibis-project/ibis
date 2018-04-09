"""
_identifiers = frozenset({
    'add',
    'aggregate',
    'all',
    'alter',
    'and',
    'as',
    'asc',
    'between',
    'by',
    'cached',
    'case',
    'cast',
    'change',
    'class',
    'column',
    'columns',
    'comment',
    'create',
    'cross',
    'data',
    'database',
    'databases',
    'date',
    'datetime',
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
    'fields',
    'fileformat',
    'first',
    'float',
    'format',
    'from',
    'full',
    'function',
    'functions',
    'group',
    'having',
    'if',
    'in',
    'inner',
    'inpath',
    'insert',
    'int',
    'integer',
    'intermediate',
    'interval',
    'into',
    'is',
    'join',
    'last',
    'left',
    'like',
    'limit',
    'lines',
    'load',
    'location',
    'metadata',
    'not',
    'null',
    'offset',
    'on',
    'or',
    'order',
    'outer',
    'partition',
    'partitioned',
    'partitions',
    'real',
    'refresh',
    'regexp',
    'rename',
    'replace',
    'returns',
    'right',
    'row',
    'schema',
    'schemas',
    'select',
    'set',
    'show',
    'stats',
    'stored',
    'string',
    'symbol',
    'table',
    'tables',
    'then',
    'to',
    'union',
    'use',
    'using',
    'values',
    'view',
    'when',
    'where',
    'with'
})
"""

# https://www.mapd.com/docs/latest/mapd-core-guide/tables/
# https://www.mapd.com/docs/latest/mapd-core-guide/views/
# https://www.mapd.com/docs/latest/mapd-core-guide/data-definition/
# https://www.mapd.com/docs/latest/mapd-core-guide/loading-data/#copy-from
_ddl = frozenset({
    'alter',  # database, table, user, view
    'array',  # copy from
    'array_delimiter',  # copy from
    'as',  # view
    'column',  # table
    'copy',  # copy from
    'create',  # database, table, user, view
    'database',  # database
    'delimiter',  # copy from
    'dictionary',  # table
    'drop',  # database, table, user, view
    'escape',  # copy from
    'exists',  # database, table, view
    'fragment_size',  # `with` clause
    'header',  # copy from
    'if',  # database, table, user, view
    'INSERTACCESS',  # user
    'is_super',  # user
    'key',  # table
    'line_delimiter',  # copy from
    'max_reject',  # copy from
    'max_rows',  # `with` clause
    'not',  # database, table, view
    'nulls',  # copy from
    'owner',  # database
    'page_size',  # `with` clause
    'partitions',  # `with` clause
    'password',  # user
    'plain_text',  # copy from
    'quoted',  # copy from
    'references',  # table
    'rename',  # table
    'shard_count',  # `with` clause
    'shared',  # table
    'replicated',  # table
    'table',  # table
    'threads',  # copy from
    'to',  # table
    'truncate',  # table
    'user',  # user
    'view',  # view
    'with'  # table, copy from
})

# https://www.mapd.com/docs/latest/mapd-core-guide/dml/
_dml = frozenset({
    'all',  # select, select/limit
    'and',  # logical operator
    'any',  # select
    'as',  # select
    'asc',  # select/order
    'between',  # comparison operator
    'by',  # select
    'calcite',  # explain
    'case',  # conditional expression
    'desc',  # select/order
    'distinct',  # select
    'explain',  # explain
    'first',  # select/order
    'from',  # select
    'group',  # select
    'having',  # select
    'ilike',  # comparison operator
    'in',  # subquery expression
    'insert',  # insert
    'is',  # comparison operator
    'join',  # select
    'last',  # select/order
    'left',  # select/join
    'like',  # comparison operator
    'limit',  # select
    # 'logicalaggregate',  # explain
    # 'logicalcalc',  # explain
    # 'logicalchi',  # explain
    # 'logicalcorrelate',  # explain
    # 'logicaldelta',  # explain
    # 'logicalexchange',  # explain
    # 'logicalfilter',  # explain
    # 'logicalintersect',  # explain
    # 'logicaljoin',  # explain
    # 'logicalmatch',  # explain
    # 'logicalminus',  # explain
    # 'logicalproject',  # explain
    # 'logicalsort',  # explain
    # 'logicaltablefunctionscan',  # explain
    # 'logicaltablemodify',  # explain
    # 'logicaltablescan',  # explain
    # 'logicalunion',  # explain
    # 'logicalvalues',  # explain
    # 'logicalwindow',  # explain
    'nullif',  # comparison operator
    'nulls',  # select/order
    'not',  # logical operator
    'offset',  # select/limit
    'on',  # select/join
    'or',  # logical operator
    'order',  # select
    'regex',  # comparison operator
    'regex_like',  # comparison operator
    'rows',  # select/limit
    'select',  # select
    'values',  # insert
    'then',  # conditional expression
    'when',  # conditional expression
    'where',  # select
    'with'  # select
})

# https://www.mapd.com/docs/latest/mapd-core-guide/tables/
# https://www.mapd.com/docs/latest/mapd-core-guide/fixed-encoding/
_data_type = frozenset({
    'bigint',
    'boolean',
    'char',
    'date',
    'decimal',
    'dict',  # encoding
    'double',
    'encoding',
    'fixed',  # encoding
    'int',
    'integer'
    'float',
    'none',  # encoding
    'null',
    'numeric',
    'precision',  # double precision
    'real',
    'smallint',
    'str',
    'text',
    'time',
    'timestamp',
    'varchar',
})

_identifiers = _ddl | _dml | _data_type


def quote_identifier(name, quotechar='"', force=False):
    if force or name.count(' ') or name in _identifiers:
        return '{0}{1}{0}'.format(quotechar, name)
    else:
        return name
