from six import StringIO
import datetime
from operator import add, mul, sub

import ibis
import ibis.expr.analysis as L
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.operations as ops

import ibis.sql.compiler as comp
import ibis.sql.transforms as transforms

import ibis.impala.identifiers as identifiers

import ibis.common as com
import ibis.util as util


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
        ops.CrossJoin: 'CROSS JOIN'
    }

    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname

    def _quote_identifier(self, name):
        return quote_identifier(name)


# ---------------------------------------------------------------------
# Scalar and array expression formatting

_sql_type_names = {
    'int8': 'tinyint',
    'int16': 'smallint',
    'int32': 'int',
    'int64': 'bigint',
    'float': 'float',
    'double': 'double',
    'string': 'string',
    'boolean': 'boolean',
    'timestamp': 'timestamp',
    'decimal': 'decimal',
}


def _cast(translator, expr):
    op = expr.op()
    arg, target_type = op.args
    arg_formatted = translator.translate(arg)

    if isinstance(arg, ir.CategoryValue) and target_type == 'int32':
        return arg_formatted
    if (
        isinstance(arg, (ir.TimeValue, ir.DateValue, ir.TimestampValue)) and
            target_type == 'int64'
            ):
        return '1000000 * unix_timestamp({})'.format(arg_formatted)
    else:
        sql_type = _type_to_sql_string(target_type)
        return 'CAST({} AS {})'.format(arg_formatted, sql_type)


def _type_to_sql_string(tval):
    if isinstance(tval, dt.Decimal):
        return 'decimal({}, {})'.format(tval.precision, tval.scale)
    name = tval.name.lower()
    try:
        return _sql_type_names[name]
    except KeyError:
        raise com.UnsupportedBackendType(name)


def _between(translator, expr):
    op = expr.op()
    comp, lower, upper = [translator.translate(x) for x in op.args]
    return '{} BETWEEN {} AND {}'.format(comp, lower, upper)


def _is_null(translator, expr):
    formatted_arg = translator.translate(expr.op().args[0])
    return '{} IS NULL'.format(formatted_arg)


def _not_null(translator, expr):
    formatted_arg = translator.translate(expr.op().args[0])
    return '{} IS NOT NULL'.format(formatted_arg)


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
    win = (win.group_by(window._group_by)
           .order_by(window._order_by))

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
    ns=.001
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
    expr : float or expression of intervals, i.e.
      1 * ibis.day() + 5 * ibis.hour()

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
                "Expected preceding values of week(), " +
                "day(), hour(), minute(), second(), millisecond(), " +
                "microseconds(), nanoseconds(); got {}".format(expr)
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
            "Expected 1 order-by variable, got {}".format(
                len(order_by_vars)
            )
        )

    order_var = window._order_by[0].op().args[0]
    timestamp_order_var = order_var.cast('int64')
    window = window._replace(order_by=timestamp_order_var,
                             how='range')

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

    _require_order_by = (ops.Lag,
                         ops.Lead,
                         ops.DenseRank,
                         ops.MinRank,
                         ops.FirstValue,
                         ops.LastValue,
                         ops.PercentRank,
                         ops.NTile,)

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
    if (isinstance(window_op, _require_order_by) and
            len(window._order_by) == 0):
        window = window.order_by(window_op.args[0])

    # Time ranges need to be converted to microseconds.
    if window.how == 'range':
        order_by_types = [type(x.op().args[0]) for x in window._order_by]
        time_range_types = (ir.TimeColumn, ir.DateColumn, ir.TimestampColumn)
        if any(col_type in time_range_types for col_type in order_by_types):
            window = _time_range_to_range_window(translator, window)

    window_formatted = _format_window(translator, window)

    arg_formatted = translator.translate(arg)
    result = '{} {}'.format(arg_formatted, window_formatted)

    if type(window_op) in _expr_transforms:
        return _expr_transforms[type(window_op)](result)
    else:
        return result


def _format_window(translator, window):
    components = []

    if len(window._group_by) > 0:
        partition_args = [translator.translate(x)
                          for x in window._group_by]
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

    def _prec(p):
        return '{} PRECEDING'.format(p) if p > 0 else 'CURRENT ROW'

    def _foll(f):
        return '{} FOLLOWING'.format(f) if f > 0 else 'CURRENT ROW'

    if p is not None and f is not None:
        frame = '{} BETWEEN {} AND {}'.format(
            window.how.upper(), _prec(p), _foll(f))

    elif p is not None:
        if isinstance(p, tuple):
            start, end = p
            frame = '{} BETWEEN {} AND {}'.format(
                window.how.upper(), _prec(start), _prec(end))
        else:
            kind = 'ROWS' if p > 0 else 'RANGE'
            frame = '{} BETWEEN {} AND UNBOUNDED FOLLOWING'.format(
                kind, _prec(p)
            )
    elif f is not None:
        if isinstance(f, tuple):
            start, end = f
            frame = '{} BETWEEN {} AND {}'.format(
                window.how.upper(), _foll(start), _foll(end))
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


def _negate(translator, expr):
    arg = expr.op().args[0]
    formatted_arg = translator.translate(arg)
    if isinstance(expr, ir.BooleanValue):
        return _not(translator, expr)
    else:
        if _needs_parens(arg):
            formatted_arg = _parenthesize(formatted_arg)
        return '-{}'.format(formatted_arg)


def _not(translator, expr):
    arg, = expr.op().args
    formatted_arg = translator.translate(arg)
    if _needs_parens(arg):
        formatted_arg = _parenthesize(formatted_arg)
    return 'NOT {}'.format(formatted_arg)


_parenthesize = '({})'.format


def unary(func_name):
    return fixed_arity(func_name, 1)


def _reduction_format(translator, func_name, arg, args, where):
    if where is not None:
        arg = where.ifelse(arg, ibis.NA)

    return '{}({})'.format(
        func_name, ', '.join(map(translator.translate, [arg] + list(args)))
    )


def _reduction(func_name):
    def formatter(translator, expr):
        op = expr.op()

        # HACK: support trailing arguments
        where = op.where
        args = [arg for arg in op.args if arg is not where]

        return _reduction_format(
            translator, func_name, args[0], args[1:], where
        )
    return formatter


def _variance_like(func_name):
    func_names = {
        'sample': '{}_samp'.format(func_name),
        'pop': '{}_pop'.format(func_name)
    }

    def formatter(translator, expr):
        arg, how, where = expr.op().args
        return _reduction_format(translator, func_names[how], arg, [], where)
    return formatter


def fixed_arity(func_name, arity):

    def formatter(translator, expr):
        op = expr.op()
        if arity != len(op.args):
            raise com.IbisError('incorrect number of args')
        return _format_call(translator, func_name, *op.args)

    return formatter


def _ifnull_workaround(translator, expr):
    op = expr.op()
    a, b = op.args

    # work around per #345, #360
    if (isinstance(a, ir.DecimalValue) and
            isinstance(b, ir.IntegerValue)):
        b = b.cast(a.type())

    return _format_call(translator, 'isnull', a, b)


def _format_call(translator, func, *args):
    formatted_args = []
    for arg in args:
        fmt_arg = translator.translate(arg)
        formatted_args.append(fmt_arg)

    return '{}({})'.format(func, ', '.join(formatted_args))


def _binary_infix_op(infix_sym):
    def formatter(translator, expr):
        op = expr.op()

        left, right = op.args

        left_arg = translator.translate(left)
        right_arg = translator.translate(right)
        if _needs_parens(left):
            left_arg = _parenthesize(left_arg)

        if _needs_parens(right):
            right_arg = _parenthesize(right_arg)

        return '{} {} {}'.format(left_arg, infix_sym, right_arg)
    return formatter


def _xor(translator, expr):
    op = expr.op()

    left_arg = translator.translate(op.left)
    right_arg = translator.translate(op.right)

    if _needs_parens(op.left):
        left_arg = _parenthesize(left_arg)

    if _needs_parens(op.right):
        right_arg = _parenthesize(right_arg)

    return '({0} OR {1}) AND NOT ({0} AND {1})'.format(left_arg, right_arg)


def _name_expr(formatted_expr, quoted_name):
    return '{} AS {}'.format(formatted_expr, quoted_name)


def _needs_parens(op):
    if isinstance(op, ir.Expr):
        op = op.op()
    op_klass = type(op)
    # function calls don't need parens
    return (op_klass in _binary_infix_ops or
            op_klass in {ops.Negate, ops.IsNull, ops.NotNull})


def _set_literal_format(translator, expr):
    value_type = expr.type().value_type

    formatted = [translator.translate(ir.literal(x, type=value_type))
                 for x in expr.op().value]

    return _parenthesize(', '.join(formatted))


def _boolean_literal_format(translator, expr):
    value = expr.op().value
    return 'TRUE' if value else 'FALSE'


def _string_literal_format(translator, expr):
    value = expr.op().value
    return "'{}'".format(value.replace("'", "\\'"))


def _number_literal_format(translator, expr):
    value = expr.op().value
    formatted = repr(value)

    if formatted in {'nan', 'inf', '-inf'}:
        return "CAST({!r} AS DOUBLE)".format(formatted)

    return formatted


def _interval_literal_format(translator, expr):
    return 'INTERVAL {} {}'.format(expr.op().value,
                                   expr.type().resolution.upper())


def _interval_from_integer(translator, expr):
    # interval cannot be selected from impala
    op = expr.op()
    arg, unit = op.args
    arg_formatted = translator.translate(arg)

    return 'INTERVAL {} {}'.format(arg_formatted,
                                   expr.type().resolution.upper())


def _date_literal_format(translator, expr):
    value = expr.op().value
    if isinstance(value, datetime.date):
        value = value.strftime('%Y-%m-%d')

    return repr(value)


def _timestamp_literal_format(translator, expr):
    value = expr.op().value
    if isinstance(value, datetime.datetime):
        value = value.strftime('%Y-%m-%d %H:%M:%S')

    return repr(value)


def quote_identifier(name, quotechar='`', force=False):
    if force or name.count(' ') or name in identifiers.impala_identifiers:
        return '{0}{1}{0}'.format(quotechar, name)
    else:
        return name


class CaseFormatter(object):

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
    formatter = CaseFormatter(translator, op.base, op.cases, op.results,
                              op.default)
    return formatter.get_result()


def _searched_case(translator, expr):
    op = expr.op()
    formatter = CaseFormatter(translator, None, op.cases, op.results,
                              op.default)
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
        translator.translate(left), translator.translate(right))


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


def _extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.args[0])

        # This is pre-2.0 Impala-style, which did not used to support the
        # SQL-99 format extract($FIELD from expr)
        return "extract({}, '{}')".format(arg, sql_attr)
    return extract_field_formatter


def _day_of_week_index(t, expr):
    arg, = expr.op().args
    return 'pmod(dayofweek({}) - 2, 7)'.format(t.translate(arg))


def _day_of_week_name(t, expr):
    arg, = expr.op().args
    return 'dayname({})'.format(t.translate(arg))


_impala_unit_names = {
    'Y': 'Y',
    'Q': 'Q',
    'M': 'MONTH',
    'W': 'W',
    'D': 'J',
    'h': 'HH',
    'm': 'MI'
}


def _truncate(translator, expr):
    op = expr.op()
    arg, unit = op.args

    arg_formatted = translator.translate(arg)
    try:
        unit = _impala_unit_names[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            '{!r} unit is not supported in timestamp truncate'.format(unit)
        )

    return "trunc({}, '{}')".format(arg_formatted, unit)


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
        return _format_call(translator, func_name, *op.arg)
    return varargs_formatter


def _substring(translator, expr):
    op = expr.op()
    arg, start, length = op.args
    arg_formatted = translator.translate(arg)
    start_formatted = translator.translate(start)

    # Impala is 1-indexed
    if length is None or isinstance(length.op(), ops.Literal):
        lvalue = length.op().value if length is not None else None
        if lvalue:
            return 'substr({}, {} + 1, {})'.format(
                arg_formatted, start_formatted, lvalue
            )
        else:
            return 'substr({}, {} + 1)'.format(arg_formatted, start_formatted)
    else:
        length_formatted = translator.translate(length)
        return 'substr({}, {} + 1, {})'.format(
            arg_formatted, start_formatted, length_formatted
        )


def _string_find(translator, expr):
    op = expr.op()
    arg, substr, start, _ = op.args
    arg_formatted = translator.translate(arg)
    substr_formatted = translator.translate(substr)

    if start is not None and not isinstance(start.op(), ops.Literal):
        start_fmt = translator.translate(start)
        return 'locate({}, {}, {} + 1) - 1'.format(
            substr_formatted, arg_formatted, start_fmt
        )
    elif start is not None and start.op().value:
        sval = start.op().value
        return 'locate({}, {}, {}) - 1'.format(
            substr_formatted, arg_formatted, sval + 1
        )
    else:
        return 'locate({}, {}) - 1'.format(substr_formatted, arg_formatted)


def _string_join(translator, expr):
    op = expr.op()
    arg, strings = op.args
    return _format_call(translator, 'concat_ws', arg, *strings)


def _parse_url(translator, expr):
    op = expr.op()

    arg, extract, key = op.args
    arg_formatted = translator.translate(arg)

    if key is None:
        return "parse_url({}, '{}')".format(arg_formatted, extract)
    else:
        key_fmt = translator.translate(key)
        return "parse_url({}, '{}', {})".format(
            arg_formatted, extract, key_fmt
        )


def _find_in_set(translator, expr):
    op = expr.op()

    arg, str_list = op.args
    arg_formatted = translator.translate(arg)
    str_formatted = ','.join([x._arg.value for x in str_list])
    return "find_in_set({}, '{}') - 1".format(arg_formatted, str_formatted)


def _round(translator, expr):
    op = expr.op()
    arg, digits = op.args

    arg_formatted = translator.translate(arg)

    if digits is not None:
        digits_formatted = translator.translate(digits)
        return 'round({}, {})'.format(arg_formatted, digits_formatted)
    return 'round({})'.format(arg_formatted)


def _hash(translator, expr):
    op = expr.op()
    arg, how = op.args

    arg_formatted = translator.translate(arg)

    if how == 'fnv':
        return 'fnv_hash({})'.format(arg_formatted)
    else:
        raise NotImplementedError(how)


def _log(translator, expr):
    op = expr.op()
    arg, base = op.args
    arg_formatted = translator.translate(arg)

    if base is None:
        return 'ln({})'.format(arg_formatted)

    base_formatted = translator.translate(base)
    return 'log({}, {})'.format(base_formatted, arg_formatted)


def _count_distinct(translator, expr):
    arg, where = expr.op().args

    if where is not None:
        arg_formatted = translator.translate(where.ifelse(arg, None))
    else:
        arg_formatted = translator.translate(arg)
    return 'count(DISTINCT {})'.format(arg_formatted)


def _literal(translator, expr):
    if isinstance(expr, ir.BooleanValue):
        typeclass = 'boolean'
    elif isinstance(expr, ir.StringValue):
        typeclass = 'string'
    elif isinstance(expr, ir.NumericValue):
        typeclass = 'number'
    elif isinstance(expr, ir.DateValue):
        typeclass = 'date'
    elif isinstance(expr, ir.TimestampValue):
        typeclass = 'timestamp'
    elif isinstance(expr, ir.IntervalValue):
        typeclass = 'interval'
    elif isinstance(expr, ir.SetValue):
        typeclass = 'set'
    else:
        raise NotImplementedError

    return _literal_formatters[typeclass](translator, expr)


def _null_literal(translator, expr):
    return 'NULL'


_literal_formatters = {
    'boolean': _boolean_literal_format,
    'number': _number_literal_format,
    'string': _string_literal_format,
    'interval': _interval_literal_format,
    'timestamp': _timestamp_literal_format,
    'date': _date_literal_format,
    'set': _set_literal_format
}


def _value_list(translator, expr):
    op = expr.op()
    formatted = [translator.translate(x) for x in op.values]
    return _parenthesize(', '.join(formatted))


def _identical_to(translator, expr):
    op = expr.op()
    if op.args[0].equals(op.args[1]):
        return 'TRUE'

    left_expr = op.left
    right_expr = op.right
    left = translator.translate(left_expr)
    right = translator.translate(right_expr)

    if _needs_parens(left_expr):
        left = _parenthesize(left)
    if _needs_parens(right_expr):
        right = _parenthesize(right)
    return '{} IS NOT DISTINCT FROM {}'.format(left, right)


_subtract_one = '({} - 1)'.format


_expr_transforms = {
    ops.RowNumber: _subtract_one,
    ops.DenseRank: _subtract_one,
    ops.MinRank: _subtract_one,
    ops.NTile: _subtract_one,
}


_binary_infix_ops = {
    # Binary operations
    ops.Add: _binary_infix_op('+'),
    ops.Subtract: _binary_infix_op('-'),
    ops.Multiply: _binary_infix_op('*'),
    ops.Divide: _binary_infix_op('/'),
    ops.Power: fixed_arity('pow', 2),
    ops.Modulus: _binary_infix_op('%'),

    # Comparisons
    ops.Equals: _binary_infix_op('='),
    ops.NotEquals: _binary_infix_op('!='),
    ops.GreaterEqual: _binary_infix_op('>='),
    ops.Greater: _binary_infix_op('>'),
    ops.LessEqual: _binary_infix_op('<='),
    ops.Less: _binary_infix_op('<'),
    ops.IdenticalTo: _identical_to,

    # Boolean comparisons
    ops.And: _binary_infix_op('AND'),
    ops.Or: _binary_infix_op('OR'),
    ops.Xor: _xor,
}


def _string_like(translator, expr):
    arg, pattern, _ = expr.op().args
    return '{} LIKE {}'.format(
        translator.translate(arg),
        translator.translate(pattern),
    )


def _sign(translator, expr):
    arg, = expr.op().args
    translated_arg = translator.translate(arg)
    translated_type = _type_to_sql_string(expr.type())
    if expr.type() != dt.float:
        return 'CAST(sign({}) AS {})'.format(translated_arg, translated_type)
    return 'sign({})'.format(translated_arg)


_operation_registry = {
    # Unary operations
    ops.NotNull: _not_null,
    ops.IsNull: _is_null,
    ops.Negate: _negate,
    ops.Not: _not,

    ops.IsNan: unary('is_nan'),
    ops.IsInf: unary('is_inf'),

    ops.IfNull: _ifnull_workaround,
    ops.NullIf: fixed_arity('nullif', 2),

    ops.ZeroIfNull: unary('zeroifnull'),
    ops.NullIfZero: unary('nullifzero'),

    ops.Abs: unary('abs'),
    ops.BaseConvert: fixed_arity('conv', 3),
    ops.Ceil: unary('ceil'),
    ops.Floor: unary('floor'),
    ops.Exp: unary('exp'),
    ops.Round: _round,

    ops.Sign: _sign,
    ops.Sqrt: unary('sqrt'),

    ops.Hash: _hash,

    ops.Log: _log,
    ops.Ln: unary('ln'),
    ops.Log2: unary('log2'),
    ops.Log10: unary('log10'),

    ops.DecimalPrecision: unary('precision'),
    ops.DecimalScale: unary('scale'),

    # Unary aggregates
    ops.CMSMedian: _reduction('appx_median'),
    ops.HLLCardinality: _reduction('ndv'),
    ops.Mean: _reduction('avg'),
    ops.Sum: _reduction('sum'),
    ops.Max: _reduction('max'),
    ops.Min: _reduction('min'),

    ops.StandardDev: _variance_like('stddev'),
    ops.Variance: _variance_like('var'),

    ops.GroupConcat: _reduction('group_concat'),

    ops.Count: _reduction('count'),
    ops.CountDistinct: _count_distinct,

    # string operations
    ops.StringLength: unary('length'),
    ops.StringAscii: unary('ascii'),
    ops.Lowercase: unary('lower'),
    ops.Uppercase: unary('upper'),
    ops.Reverse: unary('reverse'),
    ops.Strip: unary('trim'),
    ops.LStrip: unary('ltrim'),
    ops.RStrip: unary('rtrim'),
    ops.Capitalize: unary('initcap'),
    ops.Substring: _substring,
    ops.StrRight: fixed_arity('strright', 2),
    ops.Repeat: fixed_arity('repeat', 2),
    ops.StringFind: _string_find,
    ops.Translate: fixed_arity('translate', 3),
    ops.FindInSet: _find_in_set,
    ops.LPad: fixed_arity('lpad', 3),
    ops.RPad: fixed_arity('rpad', 3),
    ops.StringJoin: _string_join,
    ops.StringSQLLike: _string_like,
    ops.RegexSearch: fixed_arity('regexp_like', 2),
    ops.RegexExtract: fixed_arity('regexp_extract', 3),
    ops.RegexReplace: fixed_arity('regexp_replace', 3),
    ops.ParseURL: _parse_url,

    # Timestamp operations
    ops.Date: unary('to_date'),
    ops.TimestampNow: lambda *args: 'now()',
    ops.ExtractYear: _extract_field('year'),
    ops.ExtractMonth: _extract_field('month'),
    ops.ExtractDay: _extract_field('day'),
    ops.ExtractHour: _extract_field('hour'),
    ops.ExtractMinute: _extract_field('minute'),
    ops.ExtractSecond: _extract_field('second'),
    ops.ExtractMillisecond: _extract_field('millisecond'),
    ops.TimestampTruncate: _truncate,
    ops.DateTruncate: _truncate,
    ops.IntervalFromInteger: _interval_from_integer,

    # Other operations
    ops.E: lambda *args: 'e()',

    ops.Literal: _literal,
    ops.NullLiteral: _null_literal,

    ops.ValueList: _value_list,

    ops.Cast: _cast,

    ops.Coalesce: varargs('coalesce'),
    ops.Greatest: varargs('greatest'),
    ops.Least: varargs('least'),

    ops.Where: fixed_arity('if', 3),

    ops.Between: _between,
    ops.Contains: _binary_infix_op('IN'),
    ops.NotContains: _binary_infix_op('NOT IN'),

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

_operation_registry.update(_binary_infix_ops)


class ImpalaExprTranslator(comp.ExprTranslator):

    _registry = _operation_registry
    context_class = ImpalaContext

    def name(self, translated, name, force=True):
        return _name_expr(translated,
                          quote_identifier(name, force=force))


class ImpalaDialect(comp.Dialect):
    translator = ImpalaExprTranslator


dialect = ImpalaDialect


compiles = ImpalaExprTranslator.compiles
rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
