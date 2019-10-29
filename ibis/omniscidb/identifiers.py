"""OmniSci reserved words."""

_ddl = frozenset(
    {
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
        'with',  # table, copy from
    }
)

_dml = frozenset(
    {
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
        'max',
        'min',
        'std',
        'count',
        'mean',
        'sum',
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
        'with',  # select
    }
)

_data_type = frozenset(
    {
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
        'integer' 'float',
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
    }
)

# @TODO: check if it is necessary
_ibis = frozenset({'literal'})

_identifiers = _ddl | _dml | _data_type | _ibis


_special_chars = (' ', '*')


def quote_identifier(name, quotechar='"', force=False):
    """Quote a word when necessary or forced.

    Returns
    -------
    string
    """
    if (
        force or any(c in name for c in _special_chars) or name in _identifiers
    ) and quotechar not in name:
        return '{0}{1}{0}'.format(quotechar, name)
    else:
        return name
