from __future__ import annotations

import sqlalchemy as sa
import sqlalchemy.dialects.mysql as mysql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.mysql.registry import operation_registry


class MySQLExprTranslator(AlchemyExprTranslator):
    # https://dev.mysql.com/doc/refman/8.0/en/spatial-function-reference.html
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Boolean: mysql.BOOLEAN,
            dt.Int8: mysql.TINYINT,
            dt.Int16: mysql.INTEGER,
            dt.Int32: mysql.INTEGER,
            dt.Int64: mysql.BIGINT,
            dt.Float16: mysql.FLOAT,
            dt.Float32: mysql.FLOAT,
            dt.Float64: mysql.DOUBLE,
            dt.String: mysql.VARCHAR,
        }
    )
    integer_to_timestamp = sa.func.from_unixtime
    native_json_type = False


rewrites = MySQLExprTranslator.rewrites


class MySQLCompiler(AlchemyCompiler):
    translator_class = MySQLExprTranslator
