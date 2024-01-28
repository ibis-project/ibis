from __future__ import annotations

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.risingwave.datatypes import RisingwaveType
from ibis.backends.risingwave.registry import operation_registry
from ibis.expr.rewrites import rewrite_sample


class RisingwaveExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _supports_tuple_syntax = True
    _dialect_name = "risingwave"

    # it does support it, but we can't use it because of support for pivot
    supports_unnest_in_select = False

    type_mapper = RisingwaveType


rewrites = RisingwaveExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
def _any_all_no_op(expr):
    return expr


class RisingwaveCompiler(AlchemyCompiler):
    translator_class = RisingwaveExprTranslator
    rewrites = AlchemyCompiler.rewrites | rewrite_sample
