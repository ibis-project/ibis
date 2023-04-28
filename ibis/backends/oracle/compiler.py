from __future__ import annotations

from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.oracle.registry import operation_registry


class OracleExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _dialect_name = "oracle"


rewrites = OracleExprTranslator.rewrites


class OracleCompiler(AlchemyCompiler):
    translator_class = OracleExprTranslator
