import sqlalchemy as sa
import sqlalchemy.dialects.mysql as mysql

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyExprTranslator

from .registry import operation_registry


def add_operation(op, translation_func):
    operation_registry[op] = translation_func


class MySQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Boolean: mysql.BOOLEAN,
            dt.Int8: mysql.TINYINT,
            dt.Int32: mysql.INTEGER,
            dt.Int64: mysql.BIGINT,
            dt.Double: mysql.DOUBLE,
            dt.Float: mysql.FLOAT,
            dt.String: mysql.VARCHAR,
        }
    )


rewrites = MySQLExprTranslator.rewrites
compiles = MySQLExprTranslator.compiles


@compiles(ops.GroupConcat)
def mysql_compiles_group_concat(t, expr):
    op = expr.op()
    arg, sep, where = op.args
    if where is not None:
        case = where.ifelse(arg, ibis.NA)
        arg = t.translate(case)
    else:
        arg = t.translate(arg)
    return sa.func.group_concat(arg.op('SEPARATOR')(t.translate(sep)))
