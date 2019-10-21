import pyodbc
import sqlalchemy as sa
import sqlalchemy.dialects.mssql as mssql


import ibis.sql.alchemy as alch
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

# used for literal translate
from ibis.sql.alchemy import (
    unary,
    fixed_arity
)


# coppied from postgresql compiler
# support for of bit columns in aggregate methods
def _reduction(func_name, cast_type='int32'):
    def reduction_compiler(t, expr):
        arg, where = expr.op().args

        if arg.type().equals(dt.boolean):
            arg = arg.cast(cast_type)

        func = getattr(sa.func, func_name)

        if where is not None:
            arg = where.ifelse(arg, None)
        return func(t.translate(arg))

    return reduction_compiler


# TODO: substr and find are copied from SQLite, we should really have a
# "base" set of SQL functions that are the most common APIs across the major
# RDBMS
def _substr(t, expr):
    f = sa.func.substring

    arg, start, length = expr.op().args

    sa_arg = t.translate(arg)
    sa_start = t.translate(start)

    if length is None:
        return f(sa_arg, sa_start + 1)
    else:
        sa_length = t.translate(length)
        return f(sa_arg, sa_start + 1, sa_length)


def _log(t, expr):
    arg, base = expr.op().args
    sa_arg = t.translate(arg)
    if base is not None:
        sa_base = t.translate(base)
        return sa.cast(
            sa.func.log(
                sa.cast(sa_arg, sa.NUMERIC), sa.cast(sa_base, sa.NUMERIC)
            ),
            t.get_sqla_type(expr.type()),
        )
    return sa.func.log(sa_arg)


_operation_registry = alch._operation_registry.copy()


_operation_registry.update({
    # aggregate methods
    ops.Max: _reduction('max'),
    ops.Min: _reduction('min'),
    ops.Sum: _reduction('sum'),
    ops.Mean: _reduction('avg', 'float64'),
    # string methods
    ops.Substring: _substr,
    # math
    ops.Log: _log,
    ops.Log2: unary(lambda x: sa.func.log(x, 2)),
    ops.Log10: unary(lambda x: sa.func.log10(x)),
    ops.Ln: unary(lambda x: sa.func.log(x)),
    ops.Sin: unary(lambda x: sa.func.sin(x)),
    ops.Cos: unary(lambda x: sa.func.cos(x)),
    ops.Tan: unary(lambda x: sa.func.tan(x)),
    ops.Asin: unary(lambda x: sa.func.asin(x)),
    ops.Acos: unary(lambda x: sa.func.acos(x)),
    ops.Atan: unary(lambda x: sa.func.atan(x)),
    ops.Ceil: unary(lambda x: sa.func.ceiling(x)),
    ops.Power: fixed_arity(sa.func.power, 2),
})


class MSSQLExprTranslator(alch.AlchemyExprTranslator):
    _registry = _operation_registry
    _rewrites = alch.AlchemyExprTranslator._rewrites.copy()
    _type_map = alch.AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Boolean: pyodbc.SQL_BIT,
            dt.Int8: mssql.TINYINT,
            dt.Int32: mssql.INTEGER,
            dt.Int64: mssql.BIGINT,
            dt.Double: mssql.REAL,
            dt.Float: mssql.REAL,
            dt.String: mssql.VARCHAR,
        }
    )


rewrites = MSSQLExprTranslator.rewrites
compiles = MSSQLExprTranslator.compiles


class MSSQLDialect(alch.AlchemyDialect):

    translator = MSSQLExprTranslator


dialect = MSSQLDialect
