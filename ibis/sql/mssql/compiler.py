import pyodbc
import sqlalchemy.dialects.mssql as mssql

import ibis.sql.alchemy as alch
import ibis.expr.datatypes as dt


class MSSQLExprTranslator(alch.AlchemyExprTranslator):
    # _registry = _operation_registry
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
