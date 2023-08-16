from __future__ import annotations

import sqlalchemy as sa

from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.exasol import registry
from ibis.backends.exasol.datatypes import ExasolSQLType


class ExasolExprTranslator(AlchemyExprTranslator):
    _registry = registry.create()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _integer_to_timestamp = sa.func.from_unixtime
    _dialect_name = "exa.websocket"
    native_json_type = False
    type_mapper = ExasolSQLType


rewrites = ExasolExprTranslator.rewrites


class ExasolCompiler(AlchemyCompiler):
    translator_class = ExasolExprTranslator
    support_values_syntax_in_select = False
