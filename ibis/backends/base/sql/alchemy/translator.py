import sqlalchemy as sa

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base_sqlalchemy.compiler import ExprTranslator, QueryContext

from .datatypes import ibis_type_to_sqla, to_sqla_type
from .query_builder import to_sqlalchemy
from .registry import fixed_arity, sqlalchemy_operation_registry


class AlchemyContext(QueryContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._table_objects = {}

    def collapse(self, queries):
        if isinstance(queries, str):
            return queries

        if len(queries) > 1:
            raise NotImplementedError(
                'Only a single query is supported for SQLAlchemy backends'
            )
        return queries[0]

    def subcontext(self):
        return type(self)(
            dialect=self.dialect, parent=self, params=self.params
        )

    def _to_sql(self, expr, ctx):
        return to_sqlalchemy(expr, ctx)

    def _compile_subquery(self, expr):
        sub_ctx = self.subcontext()
        return self._to_sql(expr, sub_ctx)

    def has_table(self, expr, parent_contexts=False):
        key = self._get_table_key(expr)
        return self._key_in(
            key, '_table_objects', parent_contexts=parent_contexts
        )

    def set_table(self, expr, obj):
        key = self._get_table_key(expr)
        self._table_objects[key] = obj

    def get_table(self, expr):
        """
        Get the memoized SQLAlchemy expression object
        """
        return self._get_table_item('_table_objects', expr)


class AlchemyExprTranslator(ExprTranslator):

    _registry = sqlalchemy_operation_registry
    _rewrites = ExprTranslator._rewrites.copy()
    _type_map = ibis_type_to_sqla

    context_class = AlchemyContext

    def name(self, translated, name, force=True):
        if hasattr(translated, 'label'):
            return translated.label(name)
        return translated

    def get_sqla_type(self, data_type):
        return to_sqla_type(data_type, type_map=self._type_map)


rewrites = AlchemyExprTranslator.rewrites
compiles = AlchemyExprTranslator.compiles


@rewrites(ops.NullIfZero)
def _nullifzero(expr):
    arg = expr.op().args[0]
    return (arg == 0).ifelse(ibis.NA, arg)


@compiles(ops.Divide)
def _true_divide(t, expr):
    op = expr.op()
    left, right = args = op.args

    if util.all_of(args, ir.IntegerValue):
        return t.translate(left.div(right.cast('double')))

    return fixed_arity(lambda x, y: x / y, 2)(t, expr)


@compiles(ops.SortKey)
def _sort_key(t, expr):
    # We need to define this for window functions that have an order by
    by, ascending = expr.op().args
    sort_direction = sa.asc if ascending else sa.desc
    return sort_direction(t.translate(by))
