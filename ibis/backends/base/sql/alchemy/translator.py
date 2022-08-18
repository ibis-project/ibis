from __future__ import annotations

import sqlalchemy as sa

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base.sql.alchemy.datatypes import (
    ibis_type_to_sqla,
    to_sqla_type,
)
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    sqlalchemy_operation_registry,
)
from ibis.backends.base.sql.compiler import ExprTranslator, QueryContext


class AlchemyContext(QueryContext):
    def collapse(self, queries):
        if isinstance(queries, str):
            return queries

        if len(queries) > 1:
            raise NotImplementedError(
                'Only a single query is supported for SQLAlchemy backends'
            )
        return queries[0]

    def subcontext(self):
        return self.__class__(
            compiler=self.compiler,
            parent=self,
            params=self.params,
        )


class AlchemyExprTranslator(ExprTranslator):

    _registry = sqlalchemy_operation_registry
    _rewrites = ExprTranslator._rewrites.copy()
    _type_map = ibis_type_to_sqla

    context_class = AlchemyContext

    _bool_aggs_need_cast_to_int32 = True
    _has_reduction_filter_syntax = False

    integer_to_timestamp = sa.func.to_timestamp

    def name(self, translated, name, force=True):
        return translated.label(name)

    def get_sqla_type(self, data_type):
        return to_sqla_type(data_type, type_map=self._type_map)

    def _maybe_cast_bool(self, op, arg):
        if (
            self._bool_aggs_need_cast_to_int32
            and isinstance(op, (ops.Sum, ops.Mean, ops.Min, ops.Max))
            and isinstance(type := arg.type(), dt.Boolean)
        ):
            return arg.cast(dt.Int32(nullable=type.nullable))
        return arg

    def _reduction(self, sa_func, expr):
        op = expr.op()

        argtuple = (
            self._maybe_cast_bool(op, arg)
            for name, arg in zip(op.argnames, op.args)
            if isinstance(arg, ir.Expr) and name != "where"
        )
        if (where := op.where) is not None:
            if self._has_reduction_filter_syntax:
                sa_args = tuple(map(self.translate, argtuple))
                return sa_func(*sa_args).filter(self.translate(where))
            else:
                sa_args = tuple(
                    self.translate(where.ifelse(arg, None)) for arg in argtuple
                )
        else:
            sa_args = tuple(map(self.translate, argtuple))

        return sa_func(*sa_args)


rewrites = AlchemyExprTranslator.rewrites


@rewrites(ops.NullIfZero)
def _nullifzero(expr):
    arg = expr.op().args[0]
    return (arg == 0).ifelse(ibis.NA, arg)


# TODO This was previously implemented with the legacy `@compiles` decorator.
# This definition should now be in the registry, but there is some magic going
# on that things fail if it's not defined here (and in the registry
# `operator.truediv` is used.
def _true_divide(t, expr):
    op = expr.op()
    left, right = args = op.args

    if util.all_of(args, ir.IntegerValue):
        return t.translate(left.div(right.cast('double')))

    return fixed_arity(lambda x, y: x / y, 2)(t, expr)


AlchemyExprTranslator._registry[ops.Divide] = _true_divide
