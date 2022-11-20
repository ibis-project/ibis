import sqlalchemy as sa
from sqlalchemy.dialects import mssql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.mssql.registry import operation_registry


class MsSqlExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Boolean: mssql.BIT,
            dt.Int8: mssql.TINYINT,
            dt.Int16: mssql.SMALLINT,
            dt.Int32: mssql.INTEGER,
            dt.Int64: mssql.BIGINT,
            dt.Float16: mssql.FLOAT,
            dt.Float32: mssql.FLOAT,
            dt.Float64: mssql.REAL,
            dt.String: mssql.NVARCHAR,
        }
    )
    _bool_aggs_need_cast_to_int32 = True
    integer_to_timestamp = sa.func.from_unixtime
    native_json_type = False


rewrites = MsSqlExprTranslator.rewrites


class MsSqlCompiler(AlchemyCompiler):
    translator_class = MsSqlExprTranslator
