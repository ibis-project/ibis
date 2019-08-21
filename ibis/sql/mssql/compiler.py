import ibis.sql.alchemy as alch


class MSSQLExprTranslator(alch.AlchemyExprTranslator):
    pass
    # _registry = _operation_registry
    # _rewrites = alch.AlchemyExprTranslator._rewrites.copy()
    # _type_map = alch.AlchemyExprTranslator._type_map.copy()
    # _type_map.update(
    #     {
    #         dt.Boolean: mysql.BOOLEAN,
    #         dt.Int8: mysql.TINYINT,
    #         dt.Int32: mysql.INTEGER,
    #         dt.Int64: mysql.BIGINT,
    #         dt.Double: mysql.DOUBLE,
    #         dt.Float: mysql.FLOAT,
    #         dt.String: mysql.VARCHAR,
    #     }
    # )


rewrites = MSSQLExprTranslator.rewrites
compiles = MSSQLExprTranslator.compiles


class MSSQLDialect(alch.AlchemyDialect):

    translator = MSSQLExprTranslator


dialect = MSSQLDialect
