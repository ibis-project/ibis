from io import StringIO
from operator import add, mul, sub
from typing import Optional

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as L
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.sql.compiler as comp
import ibis.sql.transforms as transforms
import ibis.util as util
from ibis.backends.base_sql import (
    BaseExprTranslator,
    binary_infix_op,
    binary_infix_ops,
    fixed_arity,
    format_call,
    literal,
    operation_registry,
    parenthesize,
    quote_identifier,
    type_to_sql_string,
    unary,
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


def _cast(translator, expr):
    op = expr.op()
    arg, target_type = op.args
    arg_formatted = translator.translate(arg)

    if isinstance(arg, ir.CategoryValue) and target_type == dt.int32:
        return arg_formatted
    if isinstance(arg, ir.TemporalValue) and target_type == dt.int64:
        return '1000000 * unix_timestamp({})'.format(arg_formatted)
    else:
        sql_type = type_to_sql_string(target_type)
        return 'CAST({} AS {})'.format(arg_formatted, sql_type)


def _between(translator, expr):
    op = expr.op()
    comp, lower, upper = [translator.translate(x) for x in op.args]
    return '{} BETWEEN {} AND {}'.format(comp, lower, upper)


_cumulative_to_reduction = {
    ops.CumulativeSum: ops.Sum,
    ops.CumulativeMin: ops.Min,
    ops.CumulativeMax: ops.Max,
    ops.CumulativeMean: ops.Mean,
    ops.CumulativeAny: ops.Any,
    ops.CumulativeAll: ops.All,
}


def _cumulative_to_window(translator, expr, window):
    win = ibis.cumulative_window()
    win = win.group_by(window._group_by).order_by(window._order_by)

    op = expr.op()

    klass = _cumulative_to_reduction[type(op)]
    new_op = klass(*op.args)
    new_expr = expr._factory(new_op, name=expr._name)

    if type(new_op) in translator._rewrites:
        new_expr = translator._rewrites[type(new_op)](new_expr)

    new_expr = L.windowize_function(new_expr, win)
    return new_expr


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


def _time_range_to_range_window(translator, window):
    # Check that ORDER BY column is a single time column:
    order_by_vars = [x.op().args[0] for x in window._order_by]
    if len(order_by_vars) > 1:
        raise com.IbisInputError(
            "Expected 1 order-by variable, got {}".format(len(order_by_vars))
        )

    order_var = window._order_by[0].op().args[0]
    timestamp_order_var = order_var.cast('int64')
    window = window._replace(order_by=timestamp_order_var, how='range')

    # Need to change preceding interval expression to scalars
    preceding = window.preceding
    if isinstance(preceding, ir.IntervalScalar):
        new_preceding = _replace_interval_with_scalar(preceding)
        window = window._replace(preceding=new_preceding)

    return window


def _window(translator, expr):
    op = expr.op()

    arg, window = op.args
    window_op = arg.op()

    _require_order_by = (
        ops.Lag,
        ops.Lead,
        ops.DenseRank,
        ops.MinRank,
        ops.FirstValue,
        ops.LastValue,
        ops.PercentRank,
        ops.NTile,
    )

    _unsupported_reductions = (
        ops.CMSMedian,
        ops.GroupConcat,
        ops.HLLCardinality,
    )

    if isinstance(window_op, _unsupported_reductions):
        raise com.UnsupportedOperationError(
            '{} is not supported in window functions'.format(type(window_op))
        )

    if isinstance(window_op, ops.CumulativeOp):
        arg = _cumulative_to_window(translator, arg, window)
        return translator.translate(arg)

    # Some analytic functions need to have the expression of interest in
    # the ORDER BY part of the window clause
    if isinstance(window_op, _require_order_by) and len(window._order_by) == 0:
        window = window.order_by(window_op.args[0])

    # Time ranges need to be converted to microseconds.
    if window.how == 'range':
        order_by_types = [type(x.op().args[0]) for x in window._order_by]
        time_range_types = (ir.TimeColumn, ir.DateColumn, ir.TimestampColumn)
        if any(col_type in time_range_types for col_type in order_by_types):
            window = _time_range_to_range_window(translator, window)

    window_formatted = _format_window(translator, op, window)

    arg_formatted = translator.translate(arg)
    result = '{} {}'.format(arg_formatted, window_formatted)

    if type(window_op) in _expr_transforms:
        return _expr_transforms[type(window_op)](result)
    else:
        return result


def _format_window(translator, op, window):
    components = []

    if window.max_lookback is not None:
        raise NotImplementedError(
            'Rows with max lookback is not implemented '
            'for Impala-based backends.'
        )

    if len(window._group_by) > 0:
        partition_args = [translator.translate(x) for x in window._group_by]
        components.append('PARTITION BY {}'.format(', '.join(partition_args)))

    if len(window._order_by) > 0:
        order_args = []
        for expr in window._order_by:
            key = expr.op()
            translated = translator.translate(key.expr)
            if not key.ascending:
                translated += ' DESC'
            order_args.append(translated)

        components.append('ORDER BY {}'.format(', '.join(order_args)))

    p, f = window.preceding, window.following

    def _prec(p: Optional[int]) -> str:
        assert p is None or p >= 0

        if p is None:
            prefix = 'UNBOUNDED'
        else:
            if not p:
                return 'CURRENT ROW'
            prefix = str(p)
        return '{} PRECEDING'.format(prefix)

    def _foll(f: Optional[int]) -> str:
        assert f is None or f >= 0

        if f is None:
            prefix = 'UNBOUNDED'
        else:
            if not f:
                return 'CURRENT ROW'
            prefix = str(f)

        return '{} FOLLOWING'.format(prefix)

    frame_clause_not_allowed = (
        ops.Lag,
        ops.Lead,
        ops.DenseRank,
        ops.MinRank,
        ops.NTile,
        ops.PercentRank,
        ops.RowNumber,
    )

    if isinstance(op.expr.op(), frame_clause_not_allowed):
        frame = None
    elif p is not None and f is not None:
        frame = '{} BETWEEN {} AND {}'.format(
            window.how.upper(), _prec(p), _foll(f)
        )

    elif p is not None:
        if isinstance(p, tuple):
            start, end = p
            frame = '{} BETWEEN {} AND {}'.format(
                window.how.upper(), _prec(start), _prec(end)
            )
        else:
            kind = 'ROWS' if p > 0 else 'RANGE'
            frame = '{} BETWEEN {} AND UNBOUNDED FOLLOWING'.format(
                kind, _prec(p)
            )
    elif f is not None:
        if isinstance(f, tuple):
            start, end = f
            frame = '{} BETWEEN {} AND {}'.format(
                window.how.upper(), _foll(start), _foll(end)
            )
        else:
            kind = 'ROWS' if f > 0 else 'RANGE'
            frame = '{} BETWEEN UNBOUNDED PRECEDING AND {}'.format(
                kind, _foll(f)
            )
    else:
        # no-op, default is full sample
        frame = None

    if frame is not None:
        components.append(frame)

    return 'OVER ({})'.format(' '.join(components))


def _shift_like(name):
    def formatter(translator, expr):
        op = expr.op()
        arg, offset, default = op.args

        arg_formatted = translator.translate(arg)

        if default is not None:
            if offset is None:
                offset_formatted = '1'
            else:
                offset_formatted = translator.translate(offset)

            default_formatted = translator.translate(default)

            return '{}({}, {}, {})'.format(
                name, arg_formatted, offset_formatted, default_formatted
            )
        elif offset is not None:
            offset_formatted = translator.translate(offset)
            return '{}({}, {})'.format(name, arg_formatted, offset_formatted)
        else:
            return '{}({})'.format(name, arg_formatted)

    return formatter


def _nth_value(translator, expr):
    op = expr.op()
    arg, rank = op.args

    arg_formatted = translator.translate(arg)
    rank_formatted = translator.translate(rank - 1)

    return 'first_value(lag({}, {}))'.format(arg_formatted, rank_formatted)


def _ntile(translator, expr):
    op = expr.op()
    arg, buckets = map(translator.translate, op.args)
    return 'ntile({})'.format(buckets)


class CaseFormatter:
    def __init__(self, translator, base, cases, results, default):
        self.translator = translator
        self.base = base
        self.cases = cases
        self.results = results
        self.default = default

        # HACK
        self.indent = 2
        self.multiline = len(cases) > 1
        self.buf = StringIO()

    def _trans(self, expr):
        return self.translator.translate(expr)

    def get_result(self):
        self.buf.seek(0)

        self.buf.write('CASE')
        if self.base is not None:
            base_str = self._trans(self.base)
            self.buf.write(' {}'.format(base_str))

        for case, result in zip(self.cases, self.results):
            self._next_case()
            case_str = self._trans(case)
            result_str = self._trans(result)
            self.buf.write('WHEN {} THEN {}'.format(case_str, result_str))

        if self.default is not None:
            self._next_case()
            default_str = self._trans(self.default)
            self.buf.write('ELSE {}'.format(default_str))

        if self.multiline:
            self.buf.write('\nEND')
        else:
            self.buf.write(' END')

        return self.buf.getvalue()

    def _next_case(self):
        if self.multiline:
            self.buf.write('\n{}'.format(' ' * self.indent))
        else:
            self.buf.write(' ')


def _simple_case(translator, expr):
    op = expr.op()
    formatter = CaseFormatter(
        translator, op.base, op.cases, op.results, op.default
    )
    return formatter.get_result()


def _searched_case(translator, expr):
    op = expr.op()
    formatter = CaseFormatter(
        translator, None, op.cases, op.results, op.default
    )
    return formatter.get_result()


def _table_array_view(translator, expr):
    ctx = translator.context
    table = expr.op().table
    query = ctx.get_compiled_expr(table)
    return '(\n{}\n)'.format(util.indent(query, ctx.indent))


# ---------------------------------------------------------------------
# Timestamp arithmetic and other functions


def _timestamp_op(func):
    def _formatter(translator, expr):
        op = expr.op()
        left, right = op.args
        formatted_left = translator.translate(left)
        formatted_right = translator.translate(right)

        if isinstance(left, (ir.TimestampScalar, ir.DateValue)):
            formatted_left = 'cast({} as timestamp)'.format(formatted_left)

        if isinstance(right, (ir.TimestampScalar, ir.DateValue)):
            formatted_right = 'cast({} as timestamp)'.format(formatted_right)

        return '{}({}, {})'.format(func, formatted_left, formatted_right)

    return _formatter


def _timestamp_diff(translator, expr):
    op = expr.op()
    left, right = op.args

    return 'unix_timestamp({}) - unix_timestamp({})'.format(
        translator.translate(left), translator.translate(right)
    )


# ---------------------------------------------------------------------
# Semi/anti-join supports


def _exists_subquery(translator, expr):
    op = expr.op()
    ctx = translator.context

    dummy = ir.literal(1).name(ir.unnamed)

    filtered = op.foreign_table.filter(op.predicates)
    expr = filtered.projection([dummy])

    subquery = ctx.get_compiled_expr(expr)

    if isinstance(op, transforms.ExistsSubquery):
        key = 'EXISTS'
    elif isinstance(op, transforms.NotExistsSubquery):
        key = 'NOT EXISTS'
    else:
        raise NotImplementedError

    return '{} (\n{}\n)'.format(key, util.indent(subquery, ctx.indent))


def _table_column(translator, expr):
    op = expr.op()
    field_name = op.name
    quoted_name = quote_identifier(field_name, force=True)

    table = op.table
    ctx = translator.context

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if translator.permit_subquery and ctx.is_foreign_expr(table):
        proj_expr = table.projection([field_name]).to_array()
        return _table_array_view(translator, proj_expr)

    if ctx.need_aliases():
        alias = ctx.get_ref(table)
        if alias is not None:
            quoted_name = '{}.{}'.format(alias, quoted_name)

    return quoted_name


def _day_of_week_index(t, expr):
    (arg,) = expr.op().args
    return 'pmod(dayofweek({}) - 2, 7)'.format(t.translate(arg))


def _day_of_week_name(t, expr):
    (arg,) = expr.op().args
    return 'dayname({})'.format(t.translate(arg))


def _timestamp_from_unix(translator, expr):
    op = expr.op()

    val, unit = op.args
    val = util.convert_unit(val, unit, 's').cast('int32')

    arg = _from_unixtime(translator, val)
    return 'CAST({} AS timestamp)'.format(arg)


def _from_unixtime(translator, expr):
    arg = translator.translate(expr)
    return 'from_unixtime({}, "yyyy-MM-dd HH:mm:ss")'.format(arg)


def varargs(func_name):
    def varargs_formatter(translator, expr):
        op = expr.op()
        return format_call(translator, func_name, *op.arg)

    return varargs_formatter


def _null_literal(translator, expr):
    return 'NULL'


def _value_list(translator, expr):
    op = expr.op()
    formatted = [translator.translate(x) for x in op.values]
    return parenthesize(', '.join(formatted))


_subtract_one = '({} - 1)'.format


_expr_transforms = {
    ops.RowNumber: _subtract_one,
    ops.DenseRank: _subtract_one,
    ops.MinRank: _subtract_one,
    ops.NTile: _subtract_one,
}


_operation_registry = {
    # Other operations
    ops.E: lambda *args: 'e()',
    ops.Literal: literal,
    ops.NullLiteral: _null_literal,
    ops.ValueList: _value_list,
    ops.Cast: _cast,
    ops.Coalesce: varargs('coalesce'),
    ops.Greatest: varargs('greatest'),
    ops.Least: varargs('least'),
    ops.Where: fixed_arity('if', 3),
    ops.Between: _between,
    ops.Contains: binary_infix_op('IN'),
    ops.NotContains: binary_infix_op('NOT IN'),
    ops.SimpleCase: _simple_case,
    ops.SearchedCase: _searched_case,
    ops.TableColumn: _table_column,
    ops.TableArrayView: _table_array_view,
    ops.DateAdd: _timestamp_op('date_add'),
    ops.DateSub: _timestamp_op('date_sub'),
    ops.DateDiff: _timestamp_op('datediff'),
    ops.TimestampAdd: _timestamp_op('date_add'),
    ops.TimestampSub: _timestamp_op('date_sub'),
    ops.TimestampDiff: _timestamp_diff,
    ops.TimestampFromUNIX: _timestamp_from_unix,
    transforms.ExistsSubquery: _exists_subquery,
    transforms.NotExistsSubquery: _exists_subquery,
    # RowNumber, and rank functions starts with 0 in Ibis-land
    ops.RowNumber: lambda *args: 'row_number()',
    ops.DenseRank: lambda *args: 'dense_rank()',
    ops.MinRank: lambda *args: 'rank()',
    ops.PercentRank: lambda *args: 'percent_rank()',
    ops.FirstValue: unary('first_value'),
    ops.LastValue: unary('last_value'),
    ops.NthValue: _nth_value,
    ops.Lag: _shift_like('lag'),
    ops.Lead: _shift_like('lead'),
    ops.WindowOp: _window,
    ops.NTile: _ntile,
    ops.DayOfWeekIndex: _day_of_week_index,
    ops.DayOfWeekName: _day_of_week_name,
}

_operation_registry.update(operation_registry)
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
