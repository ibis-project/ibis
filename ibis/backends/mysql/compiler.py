import sqlalchemy.dialects.mysql as mysql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
)

from .registry import operation_registry


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


class MySQLCompiler(AlchemyCompiler):
    translator_class = MySQLExprTranslator
