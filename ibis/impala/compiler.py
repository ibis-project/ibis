from operator import add, mul, sub

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.sql.compiler as comp
from ibis.backends.base_sql import (
    BaseExprTranslator,
    binary_infix_ops,
    operation_registry,
    quote_identifier,
)


def build_ast(expr, context):
    assert context is not None, 'context is None'
    builder = ImpalaQueryBuilder(expr, context=context)
    return builder.get_result()


def _get_query(expr, context):
    assert context is not None, 'context is None'
    ast = build_ast(expr, context)
    query = ast.queries[0]

    return query


def to_sql(expr, context=None):
    if context is None:
        context = ImpalaDialect.make_context()
    assert context is not None, 'context is None'
    query = _get_query(expr, context)
    return query.compile()


# ----------------------------------------------------------------------
# Select compilation


class ImpalaSelectBuilder(comp.SelectBuilder):
    @property
    def _select_class(self):
        return ImpalaSelect


class ImpalaQueryBuilder(comp.QueryBuilder):

    select_builder = ImpalaSelectBuilder


class ImpalaContext(comp.QueryContext):
    def _to_sql(self, expr, ctx):
        return to_sql(expr, ctx)


class ImpalaSelect(comp.Select):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    @property
    def translator(self):
        return ImpalaExprTranslator

    @property
    def table_set_formatter(self):
        return ImpalaTableSetFormatter


class ImpalaTableSetFormatter(comp.TableSetFormatter):

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

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname

    def _quote_identifier(self, name):
        return quote_identifier(name)


_map_interval_to_microseconds = dict(
    W=604800000000,
    D=86400000000,
    h=3600000000,
    m=60000000,
    s=1000000,
    ms=1000,
    us=1,
    ns=0.001,
)


_map_interval_op_to_op = {
    # Literal Intervals have two args, i.e.
    # Literal(1, Interval(value_type=int8, unit='D', nullable=True))
    # Parse both args and multipy 1 * _map_interval_to_microseconds['D']
    ops.Literal: mul,
    ops.IntervalMultiply: mul,
    ops.IntervalAdd: add,
    ops.IntervalSubtract: sub,
}


def _replace_interval_with_scalar(expr):
    """
    Good old Depth-First Search to identify the Interval and IntervalValue
    components of the expression and return a comparable scalar expression.

    Parameters
    ----------
    expr : float or expression of intervals
        For example, ``ibis.interval(days=1) + ibis.interval(hours=5)``

    Returns
    -------
    preceding : float or ir.FloatingScalar, depending upon the expr
    """
    try:
        expr_op = expr.op()
    except AttributeError:
        expr_op = None

    if not isinstance(expr, (dt.Interval, ir.IntervalValue)):
        # Literal expressions have op method but native types do not.
        if isinstance(expr_op, ops.Literal):
            return expr_op.value
        else:
            return expr
    elif isinstance(expr, dt.Interval):
        try:
            microseconds = _map_interval_to_microseconds[expr.unit]
            return microseconds
        except KeyError:
            raise ValueError(
                "Expected preceding values of week(), "
                + "day(), hour(), minute(), second(), millisecond(), "
                + "microseconds(), nanoseconds(); got {}".format(expr)
            )
    elif expr_op.args and isinstance(expr, ir.IntervalValue):
        if len(expr_op.args) > 2:
            raise com.NotImplementedError(
                "'preceding' argument cannot be parsed."
            )
        left_arg = _replace_interval_with_scalar(expr_op.args[0])
        right_arg = _replace_interval_with_scalar(expr_op.args[1])
        method = _map_interval_op_to_op[type(expr_op)]
        return method(left_arg, right_arg)

_operation_registry = {**operation_registry}
_operation_registry.update(binary_infix_ops)


class ImpalaExprTranslator(BaseExprTranslator):
    _registry = _operation_registry
    context_class = ImpalaContext


class ImpalaDialect(comp.Dialect):
    translator = ImpalaExprTranslator


dialect = ImpalaDialect


compiles = ImpalaExprTranslator.compiles
rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
