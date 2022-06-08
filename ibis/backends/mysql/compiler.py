import sqlalchemy as sa
import sqlalchemy.dialects.mysql as mysql
import toolz

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
)
from ibis.backends.base.sql.alchemy.registry import _geospatial_functions
from ibis.backends.mysql.registry import operation_registry


class MySQLExprTranslator(AlchemyExprTranslator):
    # https://dev.mysql.com/doc/refman/8.0/en/spatial-function-reference.html
    _registry = toolz.merge(operation_registry, _geospatial_functions)
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
    _bool_aggs_need_cast_to_int32 = False
    integer_to_timestamp = sa.func.from_unixtime


rewrites = MySQLExprTranslator.rewrites


class MySQLCompiler(AlchemyCompiler):
    translator_class = MySQLExprTranslator
