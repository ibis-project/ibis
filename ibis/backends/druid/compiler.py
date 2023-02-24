from __future__ import annotations

from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.druid.registry import operation_registry


class DruidExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _dialect_name = "druid"


rewrites = DruidExprTranslator.rewrites


class DruidCompiler(AlchemyCompiler):
    translator_class = DruidExprTranslator
