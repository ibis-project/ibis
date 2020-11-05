from operator import add, mul, sub

import ibis.backends.base_sqlalchemy.compiler as comp
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base_sql import binary_infix_ops, operation_registry
from ibis.backends.base_sql.compiler import (
    BaseContext,
    BaseDialect,
    BaseExprTranslator,
    BaseQueryBuilder,
    BaseSelectBuilder,
    BaseTableSetFormatter,
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
        context = BaseDialect.make_context()
    assert context is not None, 'context is None'
    query = _get_query(expr, context)
    return query.compile()


# ----------------------------------------------------------------------
# Select compilation


class ImpalaSelectBuilder(BaseSelectBuilder):
    @property
    def _select_class(self):
        return ImpalaSelect


class ImpalaQueryBuilder(BaseQueryBuilder):

    select_builder = ImpalaSelectBuilder


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


class ImpalaTableSetFormatter(BaseTableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname


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


_operation_registry = {**operation_registry, **binary_infix_ops}


class ImpalaExprTranslator(BaseExprTranslator):
    _registry = _operation_registry
    context_class = BaseContext


class ImpalaDialect(BaseDialect):
    pass


dialect = ImpalaDialect


compiles = ImpalaExprTranslator.compiles
rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
