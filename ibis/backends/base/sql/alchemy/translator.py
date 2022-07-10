from __future__ import annotations

import sqlalchemy as sa

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
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

    # TODO(kszucs): remove this method
    def get_sqla_type(self, data_type):
        return to_sqla_type(data_type, type_map=self._type_map)

    def _reduction(self, sa_func, op):
        arg = op.arg
        if (
            self._bool_aggs_need_cast_to_int32
            and isinstance(op, (ops.Sum, ops.Mean, ops.Min, ops.Max))
            and isinstance(arg.output_dtype, dt.Boolean)
        ):
            arg = ops.Cast(arg, dt.Int32(nullable=arg.output_dtype.nullable))

        if (where := op.where) is not None:
            if self._has_reduction_filter_syntax:
                return sa_func(self.translate(arg)).filter(
                    self.translate(where)
                )
            else:
                # TODO(kszucs): this should be done by `rewrites`, there are
                # multiple places in the registry where we do ad-hoc `where`
                # condition rewrites for filterable reductions which should be
                # deduplicated as well
                where = where.to_expr().ifelse(arg, None).op()
                sa_arg = self.translate(where)
        else:
            sa_arg = self.translate(arg)

        return sa_func(sa_arg)


rewrites = AlchemyExprTranslator.rewrites


@rewrites(ops.NullIfZero)
def _nullifzero(op):
    # TODO(kszucs): avoid rountripping to expr then back to op
    expr = op.arg.to_expr()
    new_expr = (expr == 0).ifelse(ibis.NA, expr)
    return new_expr.op()


# TODO This was previously implemented with the legacy `@compiles` decorator.
# This definition should now be in the registry, but there is some magic going
# on that things fail if it's not defined here (and in the registry
# `operator.truediv` is used.
def _true_divide(t, op):
    if all(isinstance(arg.output_dtype, dt.Integer) for arg in op.args):
        # TODO(kszucs): this is a rewrite, should be done in the rewrite phase
        right, left = op.right.to_expr(), op.left.to_expr()
        new_expr = left.div(right.cast('double'))
        return t.translate(new_expr.op())

    return fixed_arity(lambda x, y: x / y, 2)(t, op)


AlchemyExprTranslator._registry[ops.Divide] = _true_divide
