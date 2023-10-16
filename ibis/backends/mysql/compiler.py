from __future__ import annotations

import sqlalchemy as sa

from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.mysql.datatypes import MySQLType
from ibis.backends.mysql.registry import operation_registry
from ibis.expr.rewrites import rewrite_sample


class MySQLExprTranslator(AlchemyExprTranslator):
    # https://dev.mysql.com/doc/refman/8.0/en/spatial-function-reference.html
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _integer_to_timestamp = sa.func.from_unixtime
    native_json_type = False
    _dialect_name = "mysql"
    type_mapper = MySQLType


rewrites = MySQLExprTranslator.rewrites


class MySQLCompiler(AlchemyCompiler):
    translator_class = MySQLExprTranslator
    support_values_syntax_in_select = False
    null_limit = None
    rewrites = AlchemyCompiler.rewrites | rewrite_sample
