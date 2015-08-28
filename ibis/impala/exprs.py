# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from six import StringIO
import datetime

import ibis
import ibis.expr.analysis as L
import ibis.expr.analytics as analytics
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.expr.temporal as tempo

from ibis.sql.ddl import ExprTranslator
import ibis.sql.transforms as transforms

import ibis.impala.identifiers as identifiers

import ibis.common as com
import ibis.util as util

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
    else:
        sql_type = _type_to_sql_string(target_type)
        return 'CAST({0!s} AS {1!s})'.format(arg_formatted, sql_type)


def _type_to_sql_string(tval):
    if isinstance(tval, dt.Decimal):
        return 'decimal({0},{1})'.format(tval.precision, tval.scale)
    else:
        return _sql_type_names[tval.name()]


def _between(translator, expr):
    op = expr.op()
    comp, lower, upper = [translator.translate(x) for x in op.args]
    return '{0!s} BETWEEN {1!s} AND {2!s}'.format(comp, lower, upper)


def _is_null(translator, expr):
    formatted_arg = translator.translate(expr.op().args[0])
    return '{0!s} IS NULL'.format(formatted_arg)


def _not_null(translator, expr):
    formatted_arg = translator.translate(expr.op().args[0])
    return '{0!s} IS NOT NULL'.format(formatted_arg)


_cumulative_to_reduction = {
    ops.CumulativeSum: ops.Sum,
    ops.CumulativeMin: ops.Min,
    ops.CumulativeMax: ops.Max,
    ops.CumulativeMean: ops.Mean,
    ops.CumulativeAny: ops.Any,
    ops.CumulativeAll: ops.All,
}


def _cumulative_to_window(expr, window):
    win = ibis.cumulative_window()
    win = (win.group_by(window._group_by)
           .order_by(window._order_by))

    op = expr.op()

    klass = _cumulative_to_reduction[type(op)]
    new_op = klass(*op.args)
    new_expr = expr._factory(new_op, name=expr._name)

    if type(new_op) in _expr_rewrites:
        new_expr = _expr_rewrites[type(new_op)](new_expr)

    new_expr = L.windowize_function(new_expr, win)
    return new_expr


def _window(translator, expr):
    op = expr.op()

    arg, window = op.args
    window_op = arg.op()

    _require_order_by = (ops.Lag,
                         ops.Lead,
                         ops.DenseRank,
                         ops.MinRank,
                         ops.FirstValue,
                         ops.LastValue)

    _unsupported_reductions = (
        ops.CMSMedian,
        ops.GroupConcat,
        ops.HLLCardinality,
    )

    if isinstance(window_op, _unsupported_reductions):
        raise com.TranslationError('{0!s} is not supported in '
                                   'window functions'
                                   .format(type(window_op)))

    if isinstance(window_op, ops.CumulativeOp):
        arg = _cumulative_to_window(arg, window)
        return translator.translate(arg)

    # Some analytic functions need to have the expression of interest in
    # the ORDER BY part of the window clause
    if (isinstance(window_op, _require_order_by) and
            len(window._order_by) == 0):
        window = window.order_by(window_op.args[0])

    window_formatted = _format_window(translator, window)

    arg_formatted = translator.translate(arg)
    result = '{0} {1}'.format(arg_formatted, window_formatted)

    if type(window_op) in _expr_transforms:
        return _expr_transforms[type(window_op)](result)
    else:
        return result


def _format_window(translator, window):
    components = []

    if len(window._group_by) > 0:
        partition_args = [translator.translate(x)
                          for x in window._group_by]
        components.append('PARTITION BY {0}'.format(', '.join(partition_args)))

    if len(window._order_by) > 0:
        order_args = []
        for expr in window._order_by:
            key = expr.op()
            translated = translator.translate(key.expr)
            if not key.ascending:
                translated += ' DESC'
            order_args.append(translated)

        components.append('ORDER BY {0}'.format(', '.join(order_args)))

    p, f = window.preceding, window.following

    def _prec(p):
        return '{0} PRECEDING'.format(p) if p > 0 else 'CURRENT ROW'

    def _foll(f):
        return '{0} FOLLOWING'.format(f) if f > 0 else 'CURRENT ROW'

    if p is not None and f is not None:
        frame = ('ROWS BETWEEN {0} AND {1}'
                 .format(_prec(p), _foll(f)))
    elif p is not None:
        if isinstance(p, tuple):
            start, end = p
            frame = ('ROWS BETWEEN {0} AND {1}'
                     .format(_prec(start), _prec(end)))
        else:
            kind = 'ROWS' if p > 0 else 'RANGE'
            frame = ('{0} BETWEEN {1} AND UNBOUNDED FOLLOWING'
                     .format(kind, _prec(p)))
    elif f is not None:
        if isinstance(f, tuple):
            start, end = f
            frame = ('ROWS BETWEEN {0} AND {1}'
                     .format(_foll(start), _foll(end)))
        else:
            kind = 'ROWS' if f > 0 else 'RANGE'
            frame = ('{0} BETWEEN UNBOUNDED PRECEDING AND {1}'
                     .format(kind, _foll(f)))
    else:
        # no-op, default is full sample
        frame = None

    if frame is not None:
        components.append(frame)

    return 'OVER ({0})'.format(' '.join(components))


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

            return '{0}({1}, {2}, {3})'.format(name, arg_formatted,
                                               offset_formatted,
                                               default_formatted)
        elif offset is not None:
            offset_formatted = translator.translate(offset)
            return '{0}({1}, {2})'.format(name, arg_formatted,
                                          offset_formatted)
        else:
            return '{0}({1})'.format(name, arg_formatted)

    return formatter


def _nth_value(translator, expr):
    op = expr.op()
    arg, rank = op.args

    arg_formatted = translator.translate(arg)
    rank_formatted = translator.translate(rank - 1)

    return 'first_value(lag({0}, {1}))'.format(arg_formatted,
                                               rank_formatted)


def _negate(translator, expr):
    arg = expr.op().args[0]
    formatted_arg = translator.translate(arg)
    if isinstance(expr, ir.BooleanValue):
        return 'NOT {0!s}'.format(formatted_arg)
    else:
        if _needs_parens(arg):
            formatted_arg = _parenthesize(formatted_arg)
        return '-{0!s}'.format(formatted_arg)


def _parenthesize(what):
    return '({0!s})'.format(what)


def _unary_op(func_name):
    return _fixed_arity_call(func_name, 1)


def _reduction_format(translator, func_name, arg, where):
    if where is not None:
        case = where.ifelse(arg, ibis.NA)
        arg = translator.translate(case)
    else:
        arg = translator.translate(arg)

    return '{0!s}({1!s})'.format(func_name, arg)


def _reduction(func_name):
    def formatter(translator, expr):
        op = expr.op()

        # HACK: support trailing arguments
        arg, where = op.args[:2]

        return _reduction_format(translator, func_name, arg, where)
    return formatter


def _variance_like(func_name):
    func_names = {
        'sample': func_name,
        'pop': '{0}_pop'.format(func_name)
    }

    def formatter(translator, expr):
        arg, where, how = expr.op().args
        return _reduction_format(translator, func_names[how], arg, where)
    return formatter


def _any_expand(expr):
    arg = expr.op().args[0]
    return arg.sum() > 0


def _notany_expand(expr):
    arg = expr.op().args[0]
    return arg.sum() == 0


def _all_expand(expr):
    arg = expr.op().args[0]
    t = ir.find_base_table(arg)
    return arg.sum() == t.count()


def _notall_expand(expr):
    arg = expr.op().args[0]
    t = ir.find_base_table(arg)
    return arg.sum() < t.count()


def _fixed_arity_call(func_name, arity):

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

    return '{0!s}({1!s})'.format(func, ', '.join(formatted_args))


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

        return '{0!s} {1!s} {2!s}'.format(left_arg, infix_sym, right_arg)
    return formatter


def _xor(translator, expr):
    op = expr.op()

    left_arg = translator.translate(op.left)
    right_arg = translator.translate(op.right)

    if _needs_parens(op.left):
        left_arg = _parenthesize(left_arg)

    if _needs_parens(op.right):
        right_arg = _parenthesize(right_arg)

    return ('{0} AND NOT {1}'
            .format('({0} {1} {2})'.format(left_arg, 'OR', right_arg),
                    '({0} {1} {2})'.format(left_arg, 'AND', right_arg)))


def _name_expr(formatted_expr, quoted_name):
    return '{0!s} AS {1!s}'.format(formatted_expr, quoted_name)


def _needs_parens(op):
    if isinstance(op, ir.Expr):
        op = op.op()
    op_klass = type(op)
    # function calls don't need parens
    return (op_klass in _binary_infix_ops or
            op_klass in [ops.Negate])


def _need_parenthesize_args(op):
    if isinstance(op, ir.Expr):
        op = op.op()
    op_klass = type(op)
    return (op_klass in _binary_infix_ops or
            op_klass in [ops.Negate])


def _boolean_literal_format(expr):
    value = expr.op().value
    return 'TRUE' if value else 'FALSE'


def _number_literal_format(expr):
    value = expr.op().value
    return repr(value)


def _string_literal_format(expr):
    value = expr.op().value
    return "'{0!s}'".format(value.replace("'", "\\'"))


def _timestamp_literal_format(expr):
    value = expr.op().value
    if isinstance(value, datetime.datetime):
        if value.microsecond != 0:
            raise ValueError(value)
        value = value.strftime('%Y-%m-%d %H:%M:%S')

    return "'{0!s}'".format(value)


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
            self.buf.write(' {0}'.format(base_str))

        for case, result in zip(self.cases, self.results):
            self._next_case()
            case_str = self._trans(case)
            result_str = self._trans(result)
            self.buf.write('WHEN {0} THEN {1}'.format(case_str, result_str))

        if self.default is not None:
            self._next_case()
            default_str = self._trans(self.default)
            self.buf.write('ELSE {0}'.format(default_str))

        if self.multiline:
            self.buf.write('\nEND')
        else:
            self.buf.write(' END')

        return self.buf.getvalue()

    def _next_case(self):
        if self.multiline:
            self.buf.write('\n{0}'.format(' ' * self.indent))
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


def _bucket(translator, expr):
    import operator

    op = expr.op()
    stmt = ibis.case()

    if op.closed == 'left':
        l_cmp = operator.le
        r_cmp = operator.lt
    else:
        l_cmp = operator.lt
        r_cmp = operator.le

    user_num_buckets = len(op.buckets) - 1

    bucket_id = 0
    if op.include_under:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else r_cmp
        else:
            cmp = operator.le if op.closed == 'right' else operator.lt
        stmt = stmt.when(cmp(op.arg, op.buckets[0]), bucket_id)
        bucket_id += 1

    for j, (lower, upper) in enumerate(zip(op.buckets, op.buckets[1:])):
        if (op.close_extreme and
            ((op.closed == 'right' and j == 0) or
             (op.closed == 'left' and j == (user_num_buckets - 1)))):
            stmt = stmt.when((lower <= op.arg) & (op.arg <= upper),
                             bucket_id)
        else:
            stmt = stmt.when(l_cmp(lower, op.arg) & r_cmp(op.arg, upper),
                             bucket_id)
        bucket_id += 1

    if op.include_over:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else l_cmp
        else:
            cmp = operator.lt if op.closed == 'right' else operator.le

        stmt = stmt.when(cmp(op.buckets[-1], op.arg), bucket_id)
        bucket_id += 1

    case_expr = stmt.end().name(expr._name)
    return _searched_case(translator, case_expr)


def _category_label(translator, expr):
    op = expr.op()

    stmt = op.args[0].case()
    for i, label in enumerate(op.labels):
        stmt = stmt.when(i, label)

    if op.nulls is not None:
        stmt = stmt.else_(op.nulls)

    case_expr = stmt.end().name(expr._name)
    return _simple_case(translator, case_expr)


def _table_array_view(translator, expr):
    ctx = translator.context
    table = expr.op().table
    query = ctx.get_compiled_expr(table)
    return '(\n{0}\n)'.format(util.indent(query, ctx.indent))


# ---------------------------------------------------------------------
# Timestamp arithmetic and other functions

def _timestamp_delta(translator, expr):
    op = expr.op()
    arg, offset = op.args
    formatted_arg = translator.translate(arg)
    return _timestamp_format_offset(offset, formatted_arg)


_impala_delta_functions = {
    tempo.Year: 'years_add',
    tempo.Month: 'months_add',
    tempo.Week: 'weeks_add',
    tempo.Day: 'days_add',
    tempo.Hour: 'hours_add',
    tempo.Minute: 'minutes_add',
    tempo.Second: 'seconds_add',
    tempo.Millisecond: 'milliseconds_add',
    tempo.Microsecond: 'microseconds_add',
    tempo.Nanosecond: 'nanoseconds_add'
}


def _timestamp_format_offset(offset, arg):
    f = _impala_delta_functions[type(offset)]
    return '{0}({1}, {2})'.format(f, arg, offset.n)


# ---------------------------------------------------------------------
# Semi/anti-join supports


def _exists_subquery(translator, expr):
    op = expr.op()
    ctx = translator.context

    expr = (op.foreign_table
            .filter(op.predicates)
            .projection([ir.literal(1).name(ir.unnamed)]))

    subquery = ctx.get_compiled_expr(expr)

    if isinstance(op, transforms.ExistsSubquery):
        key = 'EXISTS'
    elif isinstance(op, transforms.NotExistsSubquery):
        key = 'NOT EXISTS'
    else:
        raise NotImplementedError

    return '{0} (\n{1}\n)'.format(key, util.indent(subquery, ctx.indent))


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
            quoted_name = '{0}.{1}'.format(alias, quoted_name)

    return quoted_name


def _extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.args[0])

        # This is pre-2.0 Impala-style, which did not used to support the
        # SQL-99 format extract($FIELD from expr)
        return "extract({0!s}, '{1!s}')".format(arg, sql_attr)
    return extract_field_formatter


def _truncate(translator, expr):
    op = expr.op()

    arg = translator.translate(op.args[0])

    _impala_unit_names = {
        'M': 'MONTH',
        'D': 'J',
        'J': 'D',
        'H': 'HH'
    }

    unit = op.args[1]
    unit = _impala_unit_names.get(unit, unit)

    return "trunc({0!s}, '{1!s}')".format(arg, unit)


def _timestamp_from_unix(translator, expr):
    op = expr.op()

    val, unit = op.args

    if unit == 'ms':
        val = (val / 1000).cast('int32')
    elif unit == 'us':
        val = (val / 1000000).cast('int32')

    arg = _from_unixtime(translator, val)
    return 'CAST({0} AS timestamp)'.format(arg)


def _from_unixtime(translator, expr):
    arg = translator.translate(expr)
    return 'from_unixtime({0}, "yyyy-MM-dd HH:mm:ss")'.format(arg)


def _varargs(func_name):
    def varargs_formatter(translator, expr):
        op = expr.op()
        return _format_call(translator, func_name, *op.args)
    return varargs_formatter


def _substring(translator, expr):
    op = expr.op()
    arg, start, length = op.args
    arg_formatted = translator.translate(arg)

    start_formatted = translator.translate(start)

    # Impala is 1-indexed
    if length is None or isinstance(length.op(), ir.Literal):
        lvalue = length.op().value if length else None
        if lvalue:
            return 'substr({0}, {1} + 1, {2})'.format(arg_formatted,
                                                      start_formatted,
                                                      lvalue)
        else:
            return 'substr({0}, {1} + 1)'.format(arg_formatted,
                                                 start_formatted)
    else:
        length_formatted = translator.translate(length)
        return 'substr({0}, {1} + 1, {2})'.format(arg_formatted,
                                                  start_formatted,
                                                  length_formatted)


def _string_find(translator, expr):
    op = expr.op()
    arg, substr, start, _ = op.args
    arg_formatted = translator.translate(arg)
    substr_formatted = translator.translate(substr)

    if start and not isinstance(start.op(), ir.Literal):
        start_fmt = translator.translate(start)
        return 'locate({0}, {1}, {2} + 1) - 1'.format(substr_formatted,
                                                      arg_formatted,
                                                      start_fmt)
    elif start and start.op().value:
        sval = start.op().value
        return 'locate({0}, {1}, {2}) - 1'.format(substr_formatted,
                                                  arg_formatted,
                                                  sval + 1)
    else:
        return 'locate({0}, {1}) - 1'.format(substr_formatted, arg_formatted)


def _string_join(translator, expr):
    op = expr.op()
    arg, strings = op.args
    return _format_call(translator, 'concat_ws', arg, *strings)


def _parse_url(translator, expr):
    op = expr.op()

    arg, extract, key = op.args
    arg_formatted = translator.translate(arg)

    if key is None:
        return "parse_url({0}, '{1}')".format(arg_formatted, extract)
    elif not isinstance(key.op(), ir.Literal):
        key_fmt = translator.translate(key)
        return "parse_url({0}, '{1}', {2})".format(arg_formatted,
                                                   extract,
                                                   key_fmt)
    else:
        return "parse_url({0}, '{1}', {2})".format(arg_formatted,
                                                   extract,
                                                   key)


def _find_in_set(translator, expr):
    op = expr.op()

    arg, str_list = op.args
    arg_formatted = translator.translate(arg)
    str_formatted = ','.join([x._arg.value for x in str_list])
    return "find_in_set({0}, '{1}') - 1".format(arg_formatted, str_formatted)


def _round(translator, expr):
    op = expr.op()
    arg, digits = op.args

    arg_formatted = translator.translate(arg)

    if digits is not None:
        digits_formatted = translator.translate(digits)
        return 'round({0}, {1})'.format(arg_formatted,
                                        digits_formatted)
    else:
        return 'round({0})'.format(arg_formatted)


def _hash(translator, expr):
    op = expr.op()
    arg, how = op.args

    arg_formatted = translator.translate(arg)

    if how == 'fnv':
        return 'fnv_hash({0})'.format(arg_formatted)
    else:
        raise NotImplementedError(how)


def _log(translator, expr):
    op = expr.op()
    arg, base = op.args
    arg_formatted = translator.translate(arg)

    if base is None:
        return 'ln({0})'.format(arg_formatted)
    else:
        return 'log({0}, {1})'.format(arg_formatted,
                                      translator.translate(base))


def _count_distinct(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.args[0])
    return 'COUNT(DISTINCT {0})'.format(arg_formatted)


def _literal(translator, expr):
    if isinstance(expr, ir.BooleanValue):
        typeclass = 'boolean'
    elif isinstance(expr, ir.StringValue):
        typeclass = 'string'
    elif isinstance(expr, ir.NumericValue):
        typeclass = 'number'
    elif isinstance(expr, ir.TimestampValue):
        typeclass = 'timestamp'
    else:
        raise NotImplementedError

    return _literal_formatters[typeclass](expr)


def _null_literal(translator, expr):
    return 'NULL'


_literal_formatters = {
    'boolean': _boolean_literal_format,
    'number': _number_literal_format,
    'string': _string_literal_format,
    'timestamp': _timestamp_literal_format
}


def _value_list(translator, expr):
    op = expr.op()
    formatted = [translator.translate(x) for x in op.values]
    return '({0})'.format(', '.join(formatted))


def _not_implemented(translator, expr):
    raise NotImplementedError


_subtract_one = '{0} - 1'.format


_expr_transforms = {
    ops.RowNumber: _subtract_one,
    ops.DenseRank: _subtract_one,
    ops.MinRank: _subtract_one,
}


_expr_rewrites = {
    ops.Any: _any_expand,
    ops.All: _all_expand,
    ops.NotAny: _notany_expand,
    ops.NotAll: _notall_expand,
}


_binary_infix_ops = {
    # Binary operations
    ops.Add: _binary_infix_op('+'),
    ops.Subtract: _binary_infix_op('-'),
    ops.Multiply: _binary_infix_op('*'),
    ops.Divide: _binary_infix_op('/'),
    ops.Power: _fixed_arity_call('pow', 2),
    ops.Modulus: _binary_infix_op('%'),

    # Comparisons
    ops.Equals: _binary_infix_op('='),
    ops.NotEquals: _binary_infix_op('!='),
    ops.GreaterEqual: _binary_infix_op('>='),
    ops.Greater: _binary_infix_op('>'),
    ops.LessEqual: _binary_infix_op('<='),
    ops.Less: _binary_infix_op('<'),

    # Boolean comparisons
    ops.And: _binary_infix_op('AND'),
    ops.Or: _binary_infix_op('OR'),
    ops.Xor: _xor,
}


_operation_registry = {
    # Unary operations
    ops.NotNull: _not_null,
    ops.IsNull: _is_null,
    ops.Negate: _negate,

    ops.IfNull: _ifnull_workaround,
    ops.NullIf: _fixed_arity_call('nullif', 2),

    ops.ZeroIfNull: _unary_op('zeroifnull'),

    ops.Abs: _unary_op('abs'),
    ops.BaseConvert: _fixed_arity_call('conv', 3),
    ops.Ceil: _unary_op('ceil'),
    ops.Floor: _unary_op('floor'),
    ops.Exp: _unary_op('exp'),
    ops.Round: _round,

    ops.Sign: _unary_op('sign'),
    ops.Sqrt: _unary_op('sqrt'),

    ops.Hash: _hash,

    ops.Log: _log,
    ops.Ln: _unary_op('ln'),
    ops.Log2: _unary_op('log2'),
    ops.Log10: _unary_op('log10'),

    ops.DecimalPrecision: _unary_op('precision'),
    ops.DecimalScale: _unary_op('scale'),

    # Unary aggregates
    ops.CMSMedian: _reduction('appx_median'),
    ops.HLLCardinality: _reduction('ndv'),
    ops.Mean: _reduction('avg'),
    ops.Sum: _reduction('sum'),
    ops.Max: _reduction('max'),
    ops.Min: _reduction('min'),

    ops.StandardDev: _variance_like('stddev'),
    ops.Variance: _variance_like('variance'),

    ops.GroupConcat: _fixed_arity_call('group_concat', 2),

    ops.Count: _reduction('count'),
    ops.CountDistinct: _count_distinct,

    # string operations
    ops.StringLength: _unary_op('length'),
    ops.StringAscii: _unary_op('ascii'),
    ops.Lowercase: _unary_op('lower'),
    ops.Uppercase: _unary_op('upper'),
    ops.Reverse: _unary_op('reverse'),
    ops.Strip: _unary_op('trim'),
    ops.LStrip: _unary_op('ltrim'),
    ops.RStrip: _unary_op('rtrim'),
    ops.Capitalize: _unary_op('initcap'),
    ops.Substring: _substring,
    ops.StrRight: _fixed_arity_call('strright', 2),
    ops.Repeat: _fixed_arity_call('repeat', 2),
    ops.StringFind: _string_find,
    ops.Translate: _fixed_arity_call('translate', 3),
    ops.FindInSet: _find_in_set,
    ops.LPad: _fixed_arity_call('lpad', 3),
    ops.RPad: _fixed_arity_call('rpad', 3),
    ops.StringJoin: _string_join,
    ops.StringSQLLike: _binary_infix_op('LIKE'),
    ops.RegexSearch: _binary_infix_op('RLIKE'),
    ops.RegexExtract: _fixed_arity_call('regexp_extract', 3),
    ops.RegexReplace: _fixed_arity_call('regexp_replace', 3),
    ops.ParseURL: _parse_url,

    # Timestamp operations
    ops.TimestampNow: lambda *args: 'now()',
    ops.ExtractYear: _extract_field('year'),
    ops.ExtractMonth: _extract_field('month'),
    ops.ExtractDay: _extract_field('day'),
    ops.ExtractHour: _extract_field('hour'),
    ops.ExtractMinute: _extract_field('minute'),
    ops.ExtractSecond: _extract_field('second'),
    ops.ExtractMillisecond: _extract_field('millisecond'),
    ops.Truncate: _truncate,

    # Other operations
    ops.E: lambda *args: 'e()',

    ir.Literal: _literal,
    ir.NullLiteral: _null_literal,

    ir.ValueList: _value_list,

    ops.Cast: _cast,

    ops.Coalesce: _varargs('coalesce'),
    ops.Greatest: _varargs('greatest'),
    ops.Least: _varargs('least'),

    ops.Where: _fixed_arity_call('if', 3),

    ops.Between: _between,
    ops.Contains: _binary_infix_op('IN'),
    ops.NotContains: _binary_infix_op('NOT IN'),

    analytics.Bucket: _bucket,
    analytics.CategoryLabel: _category_label,

    ops.SimpleCase: _simple_case,
    ops.SearchedCase: _searched_case,

    ops.TableColumn: _table_column,

    ops.TableArrayView: _table_array_view,

    ops.TimestampDelta: _timestamp_delta,
    ops.TimestampFromUNIX: _timestamp_from_unix,

    transforms.ExistsSubquery: _exists_subquery,
    transforms.NotExistsSubquery: _exists_subquery,

    # RowNumber, and rank functions starts with 0 in Ibis-land
    ops.RowNumber: lambda *args: 'row_number()',
    ops.DenseRank: lambda *args: 'dense_rank()',
    ops.MinRank: lambda *args: 'rank()',

    ops.FirstValue: _unary_op('first_value'),
    ops.LastValue: _unary_op('last_value'),
    ops.NthValue: _nth_value,
    ops.Lag: _shift_like('lag'),
    ops.Lead: _shift_like('lead'),
    ops.WindowOp: _window
}

_operation_registry.update(_binary_infix_ops)


class ImpalaExprTranslator(ExprTranslator):

    _registry = _operation_registry
    _rewrites = _expr_rewrites

    def name(self, translated, name, force=True):
        return _name_expr(translated,
                          quote_identifier(name, force=force))
