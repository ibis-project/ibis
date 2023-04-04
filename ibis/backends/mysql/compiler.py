from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    to_sqla_type,
)
from ibis.backends.mysql.registry import operation_registry


class MySQLExprTranslator(AlchemyExprTranslator):
    # https://dev.mysql.com/doc/refman/8.0/en/spatial-function-reference.html
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _integer_to_timestamp = sa.func.from_unixtime
    native_json_type = False
    _dialect_name = "mysql"


rewrites = MySQLExprTranslator.rewrites


class MySQLCompiler(AlchemyCompiler):
    translator_class = MySQLExprTranslator
    support_values_syntax_in_select = False


_MYSQL_TYPE_MAP = {
    dt.Boolean: mysql.BOOLEAN,
    dt.Int8: mysql.TINYINT,
    dt.Int16: mysql.SMALLINT,
    dt.Int32: mysql.INTEGER,
    dt.Int64: mysql.BIGINT,
    dt.Float16: mysql.FLOAT,
    dt.Float32: mysql.FLOAT,
    dt.Float64: mysql.DOUBLE,
    dt.String: mysql.TEXT,
    dt.JSON: mysql.JSON,
}


@to_sqla_type.register(mysql.dialect, tuple(_MYSQL_TYPE_MAP.keys()))
def _simple_types(_, itype):
    return _MYSQL_TYPE_MAP[type(itype)]
