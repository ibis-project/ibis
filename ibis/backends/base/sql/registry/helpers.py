import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir

from . import identifiers


def format_call(translator, func, *args):
    formatted_args = []
    for arg in args:
        fmt_arg = translator.translate(arg)
        formatted_args.append(fmt_arg)

    return '{}({})'.format(func, ', '.join(formatted_args))


def quote_identifier(name, quotechar='`', force=False):
    """Add quotes to the `name` identifier if needed."""
    if force or name.count(' ') or name in identifiers.base_identifiers:
        return '{0}{1}{0}'.format(quotechar, name)
    else:
        return name


def needs_parens(op):
    if isinstance(op, ir.Expr):
        op = op.op()
    op_klass = type(op)
    # function calls don't need parens
    return op_klass in {
        ops.Negate,
        ops.IsNull,
        ops.NotNull,
        ops.Add,
        ops.Subtract,
        ops.Multiply,
        ops.Divide,
        ops.Power,
        ops.Modulus,
        ops.Equals,
        ops.NotEquals,
        ops.GreaterEqual,
        ops.Greater,
        ops.LessEqual,
        ops.Less,
        ops.IdenticalTo,
        ops.And,
        ops.Or,
        ops.Xor,
    }


parenthesize = '({})'.format


sql_type_names = {
    'int8': 'tinyint',
    'int16': 'smallint',
    'int32': 'int',
    'int64': 'bigint',
    'float': 'float',
    'float32': 'float',
    'double': 'double',
    'float64': 'double',
    'string': 'string',
    'boolean': 'boolean',
    'timestamp': 'timestamp',
    'decimal': 'decimal',
}


def type_to_sql_string(tval):
    if isinstance(tval, dt.Decimal):
        return f'decimal({tval.precision}, {tval.scale})'
    name = tval.name.lower()
    try:
        return sql_type_names[name]
    except KeyError:
        raise com.UnsupportedBackendType(name)
