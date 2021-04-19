import ibis.backends.base_sqlalchemy.compiler as comp
import ibis.expr.operations as ops
from ibis.backends.base.sql import (
    binary_infix_ops,
    operation_registry,
    quote_identifier,
)


# ----------------------------------------------------------------------
# Select compilation


class BaseSelectBuilder(comp.SelectBuilder):
    @property
    def _select_class(self):
        return BaseSelect


class BaseQueryBuilder(comp.QueryBuilder):

    select_builder = BaseSelectBuilder


class BaseContext(comp.QueryContext):
    def _to_sql(self, expr, ctx):
        from ibis.backends.base import Backend
        return Backend().compile(expr, ctx)


class BaseSelect(comp.Select):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    @property
    def translator(self):
        return BaseExprTranslator

    @property
    def table_set_formatter(self):
        return BaseTableSetFormatter


class BaseTableSetFormatter(comp.TableSetFormatter):

    _join_names = {
        ops.InnerJoin: 'INNER JOIN',
        ops.LeftJoin: 'LEFT OUTER JOIN',
        ops.RightJoin: 'RIGHT OUTER JOIN',
        ops.OuterJoin: 'FULL OUTER JOIN',
        ops.LeftAntiJoin: 'LEFT ANTI JOIN',
        ops.LeftSemiJoin: 'LEFT SEMI JOIN',
        ops.CrossJoin: 'CROSS JOIN',
    }

    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        return jname

    def _quote_identifier(self, name):
        return quote_identifier(name)


_operation_registry = {**operation_registry, **binary_infix_ops}


# TODO move the name method to comp.ExprTranslator and use that instead
class BaseExprTranslator(comp.ExprTranslator):
    """Base expression translator."""

    _registry = _operation_registry
    context_class = BaseContext

    @staticmethod
    def _name_expr(formatted_expr, quoted_name):
        return '{} AS {}'.format(formatted_expr, quoted_name)

    def name(self, translated, name, force=True):
        """Return expression with its identifier."""
        return self._name_expr(translated, quote_identifier(name, force=force))


class BaseDialect(comp.Dialect):
    translator = BaseExprTranslator


compiles = BaseExprTranslator.compiles
rewrites = BaseExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
