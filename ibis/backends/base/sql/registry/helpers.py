from __future__ import annotations

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.registry import identifiers


def format_call(translator, func, *args):
    formatted_args = []
    for arg in args:
        fmt_arg = translator.translate(arg)
        formatted_args.append(fmt_arg)

    return "{}({})".format(func, ", ".join(formatted_args))


def quote_identifier(name, quotechar="`", force=False):
    """Add quotes to the `name` identifier if needed."""
    if force or name.count(" ") or name in identifiers.base_identifiers:
        return f"{quotechar}{name}{quotechar}"
    else:
        return name


_NEEDS_PARENS_OPS = (
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
)


def needs_parens(op: ops.Node):
    if isinstance(op, ops.Alias):
        op = op.arg
    return isinstance(op, _NEEDS_PARENS_OPS)


parenthesize = "({})".format


sql_type_names = {
    "int8": "tinyint",
    "int16": "smallint",
    "int32": "int",
    "int64": "bigint",
    "float": "float",
    "float32": "float",
    "double": "double",
    "float64": "double",
    "string": "string",
    "boolean": "boolean",
    "timestamp": "timestamp",
    "decimal": "decimal",
}


def type_to_sql_string(tval):
    if tval.is_decimal():
        return f"decimal({tval.precision}, {tval.scale})"
    name = tval.name.lower()
    try:
        return sql_type_names[name]
    except KeyError:
        raise com.UnsupportedBackendType(name)
