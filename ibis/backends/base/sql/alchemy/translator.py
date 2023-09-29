from __future__ import annotations

import functools
import operator

import sqlalchemy as sa
from sqlalchemy.engine.default import DefaultDialect

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.backends.base.sql.alchemy.registry import (
    fixed_arity,
    sqlalchemy_operation_registry,
)
from ibis.backends.base.sql.compiler import ExprTranslator, QueryContext

_DEFAULT_DIALECT = DefaultDialect()


class AlchemyContext(QueryContext):
    def collapse(self, queries):
        if isinstance(queries, str):
            return queries

        if len(queries) > 1:
            raise NotImplementedError(
                "Only a single query is supported for SQLAlchemy backends"
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

    type_mapper = AlchemyType
    context_class = AlchemyContext

    _bool_aggs_need_cast_to_int32 = True
    _has_reduction_filter_syntax = False
    _supports_tuple_syntax = False
    _integer_to_timestamp = staticmethod(sa.func.to_timestamp)
    _timestamp_type = sa.TIMESTAMP

    def integer_to_timestamp(self, arg, tz: str | None = None):
        return sa.cast(
            self._integer_to_timestamp(arg),
            self._timestamp_type(timezone=tz is not None),
        )

    native_json_type = True
    _quote_column_names = None  # let the dialect decide how to quote
    _quote_table_names = None

    _require_order_by = (
        ops.DenseRank,
        ops.MinRank,
        ops.NTile,
        ops.PercentRank,
        ops.CumeDist,
    )

    _dialect_name = "default"

    supports_unnest_in_select = True

    @classmethod
    def get_sqla_type(cls, ibis_type):
        return cls.type_mapper.from_ibis(ibis_type)

    @classmethod
    def get_ibis_type(cls, sqla_type, nullable=True):
        return cls.type_mapper.to_ibis(sqla_type, nullable=nullable)

    @functools.cached_property
    def dialect(self) -> sa.engine.interfaces.Dialect:
        if (name := self._dialect_name) == "default":
            return _DEFAULT_DIALECT
        dialect_cls = sa.dialects.registry.load(name)
        return dialect_cls()

    def _schema_to_sqlalchemy_columns(self, schema):
        return [
            sa.Column(name, self.get_sqla_type(dtype), quote=self._quote_column_names)
            for name, dtype in schema.items()
        ]

    def name(self, translated, name, force=False):
        return translated.label(
            sa.sql.quoted_name(name, quote=force or self._quote_column_names)
        )

    def _maybe_cast_bool(self, op, arg):
        if (
            self._bool_aggs_need_cast_to_int32
            and isinstance(op, (ops.Sum, ops.Mean, ops.Min, ops.Max))
            and (dtype := arg.dtype).is_boolean()
        ):
            return ops.Cast(arg, dt.Int32(nullable=dtype.nullable))
        return arg

    def _reduction(self, sa_func, op):
        argtuple = (
            self._maybe_cast_bool(op, arg)
            for name, arg in zip(op.argnames, op.args)
            if isinstance(arg, ops.Node) and name != "where"
        )
        if (where := op.where) is not None:
            if self._has_reduction_filter_syntax:
                sa_args = tuple(map(self.translate, argtuple))
                return sa_func(*sa_args).filter(self.translate(where))
            else:
                sa_args = tuple(
                    self.translate(ops.IfElse(where, arg, None)) for arg in argtuple
                )
        else:
            sa_args = tuple(map(self.translate, argtuple))

        return sa_func(*sa_args)


rewrites = AlchemyExprTranslator.rewrites


# TODO This was previously implemented with the legacy `@compiles` decorator.
# This definition should now be in the registry, but there is some magic going
# on that things fail if it's not defined here (and in the registry
# `operator.truediv` is used.
def _true_divide(t, op):
    if all(arg.dtype.is_integer() for arg in op.args):
        # TODO(kszucs): this should be done in the rewrite phase
        right, left = op.right.to_expr(), op.left.to_expr()
        new_expr = left.div(right.cast(dt.double))
        return t.translate(new_expr.op())

    return fixed_arity(operator.truediv, 2)(t, op)


AlchemyExprTranslator._registry[ops.Divide] = _true_divide
