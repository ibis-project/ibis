from six import StringIO
from datetime import date, datetime

# import ibis
# import ibis.expr.analysis as L
# import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.expr.temporal as tempo

# import ibis.sql.compiler as comp
import ibis.sql.transforms as transforms

# TODO: create absolute import
from .identifiers import quote_identifier
from .types import _type_to_sql_string

import ibis.common as com
import ibis.util as util


# ---------------------------------------------------------------------
# Scalar and array expression formatting


def _cast(translator, expr):
    op = expr.op()
    arg, target = op.args
    arg_ = translator.translate(arg)

    if isinstance(arg, ir.CategoryValue) and target == 'int32':
        return arg_
    else:
        type_ = _type_to_sql_string(target)
        return 'CAST({0!s} AS {1!s})'.format(arg_, type_)


def _between(translator, expr):
    op = expr.op()
    arg, lower, upper = [translator.translate(x) for x in op.args]
    return '{0!s} BETWEEN {1!s} AND {2!s}'.format(arg, lower, upper)


# def _shift_like(name):

#     def formatter(translator, expr):
#         op = expr.op()
#         arg, offset, default = op.args

#         arg_formatted = translator.translate(arg)

#         if default is not None:
#             if offset is None:
#                 offset_formatted = '1'
#             else:
#                 offset_formatted = translator.translate(offset)

#             default_formatted = translator.translate(default)

#             return '{0}({1}, {2}, {3})'.format(name, arg_formatted,
#                                                offset_formatted,
#                                                default_formatted)
#         elif offset is not None:
#             offset_formatted = translator.translate(offset)
#             return '{0}({1}, {2})'.format(name, arg_formatted,
#                                           offset_formatted)
#         else:
#             return '{0}({1})'.format(name, arg_formatted)

#     return formatter


# def _nth_value(translator, expr):
#     op = expr.op()
#     arg, rank = op.args

#     arg_formatted = translator.translate(arg)
#     rank_formatted = translator.translate(rank - 1)

#     return 'first_value(lag({0}, {1}))'.format(arg_formatted,
#                                                rank_formatted)


# def _ntile(translator, expr):
#     op = expr.op()
#     arg, buckets = map(translator.translate, op.args)
#     return 'ntile({})'.format(buckets)


def _negate(translator, expr):
    arg = expr.op().args[0]
    formatted_arg = translator.translate(arg)
    if isinstance(expr, ir.BooleanValue):
        return 'NOT {0!s}'.format(formatted_arg)
    else:
        arg_ = _parenthesize(translator, arg)
        return '-{0!s}'.format(arg_)


def _not(translator, expr):
    return 'NOT {}'.format(*map(translator.translate, expr.op().args))


def _parenthesize(translator, expr):
    op = expr.op()
    op_klass = type(op)

    # function calls don't need parens
    what = translator.translate(expr)
    if (op_klass in _binary_infix_ops) or (op_klass in _unary_ops):
        return '({0!s})'.format(what)
    else:
        return what


def unary(func_name):
    return fixed_arity(func_name, 1)


def fixed_arity(func_name, arity):
    def formatter(translator, expr):
        op = expr.op()
        arg_count = len(op.args)
        if arity != arg_count:
            msg = 'Incorrect number of args {0} instead of {1}'
            raise com.TranslationError(msg.format(arg_count, arity))
        return _call(translator, func_name, *op.args)
    return formatter


def agg(func):
    def formatter(translator, expr):
        return _aggregate(translator, func, *expr.op().args)
    return formatter


def agg_variance_like(func):
    variants = {'sample': '{0}Samp'.format(func),
                'pop': '{0}Pop'.format(func)}

    def formatter(translator, expr):
        arg, where, how = expr.op().args
        return _aggregate(translator, variants[how], arg, where)

    return formatter


def binary_infix_op(infix_sym):
    def formatter(translator, expr):
        op = expr.op()

        left, right = op.args
        left_ = _parenthesize(translator, left)
        right_ = _parenthesize(translator, right)

        return '{0!s} {1!s} {2!s}'.format(left_, infix_sym, right_)
    return formatter


def _call(translator, func, *args):
    args_ = ', '.join(map(translator.translate, args))
    return '{0!s}({1!s})'.format(func, args_)


def _aggregate(translator, func, arg, where=None):
    if where is not None:
        return _call(translator, func + 'If', arg, where)
    else:
        return _call(translator, func, arg)


def _xor(translator, expr):
    op = expr.op()
    left_ = _parenthesize(translator, op.left)
    right_ = _parenthesize(translator, op.right)
    return 'xor({0}, {1})'.format(left_, right_)


def _name_expr(formatted_expr, quoted_name):
    return '{0!s} AS {1!s}'.format(formatted_expr, quoted_name)


def varargs(func_name):
    def varargs_formatter(translator, expr):
        op = expr.op()
        return _call(translator, func_name, *op.args)
    return varargs_formatter


def _substring(translator, expr):
    # arg_ is the formatted notation
    op = expr.op()
    arg, start, length = op.args
    arg_, start_ = translator.translate(arg), translator.translate(start)

    # Clickhouse is 1-indexed
    if length is None or isinstance(length.op(), ir.Literal):
        if length is not None:
            length_ = length.op().value
            return 'substring({0}, {1} + 1, {2})'.format(arg_, start_, length_)
        else:
            return 'substring({0}, {1} + 1)'.format(arg_, start_)
    else:
        length_ = translator.translate(length)
        return 'substring({0}, {1} + 1, {2})'.format(arg_, start_, length_)


def _string_find(translator, expr):
    op = expr.op()
    arg, substr, start, _ = op.args
    if start is not None:
        raise com.TranslationError('String find doesn\'t support start argument')

    return _call(translator, 'position', arg, substr) + ' - 1'


def _regex_extract(translator, expr):
    op = expr.op()
    arg, pattern, index = op.args
    arg_, pattern_ = translator.translate(arg), translator.translate(pattern)

    if index is not None:
        index_ = translator.translate(index)
        return 'extractAll({0}, {1})[{2} + 1]'.format(arg_, pattern_, index_)

    return 'extractAll({0}, {1})'.format(arg_, pattern_)


def _string_join(translator, expr):
    op = expr.op()
    arg, strings = op.args
    return _call(translator, 'concat_ws', arg, *strings)


# TODO
def _parse_url(translator, expr):
    op = expr.op()

    arg, extract, key = op.args
    arg_formatted = translator.translate(arg)

    if key is None:
        return "parse_url({0}, '{1}')".format(arg_formatted, extract)
    else:
        key_fmt = translator.translate(key)
        return "parse_url({0}, '{1}', {2})".format(arg_formatted,
                                                   extract, key_fmt)


def _index_of(translator, expr):
    op = expr.op()

    arg, arr = op.args
    arg_formatted = translator.translate(arg)
    arr_formatted = ','.join(map(translator.translate, arr))
    return "indexOf([{0}], {1}) - 1".format(arr_formatted, arg_formatted)


def _sign(translator, expr):
    """Workaround for missing sign function"""
    op = expr.op()
    arg, = op.args
    arg_ = translator.translate(arg)
    return 'intDivOrZero({0}, abs({0}))'.format(arg_)


def _round(translator, expr):
    op = expr.op()
    arg, digits = op.args

    arg_ = translator.translate(arg)
    if digits is not None:
        digits_ = translator.translate(digits)
        return 'round({0}, {1})'.format(arg_, digits_)
    else:
        return 'round({0})'.format(arg_)


# TODO there are a lot of hash functions in clickhouse
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

    # I don't know how to check base value properly
    if base is None:
        return 'log({0})'.format(arg_formatted)
    elif base._arg.value == 2:
        return 'log2({0})'.format(arg_formatted)
    elif base._arg.value == 10:
        return 'log10({0})'.format(arg_formatted)
    else:
        raise ValueError('Base {} for logarithm not supported!'.format(base))


def _value_list(translator, expr):
    op = expr.op()
    values_ = map(translator.translate, op.values)
    return '({0})'.format(', '.join(values_))


def literal(translator, expr):
    value = expr.op().value
    if isinstance(expr, ir.BooleanValue):
        return '1' if value else '0'
    elif isinstance(expr, ir.StringValue):
        return "'{0!s}'".format(value.replace("'", "\\'"))
    elif isinstance(expr, ir.NumericValue):
        return repr(value)
    elif isinstance(expr, ir.TimestampValue):
        if isinstance(value, datetime):
            if value.microsecond != 0:
                msg = 'Unspoorted subsecond accuracy {}'
                raise ValueError(msg.format(value))
            value = value.strftime('%Y-%m-%d %H:%M:%S')
        return "toDateTime('{0!s}')".format(value)
    elif isinstance(expr, ir.DateValue):
        if isinstance(value, date):
            value = value.strftime('%Y-%m-%d')
        return "toDate('{0!s}')".format(value)
    else:
        raise NotImplementedError


class CaseFormatter(object):

    def __init__(self, translator, base, cases, results, default):
        self.translator = translator
        self.base = base
        self.cases = cases
        self.results = results

        if default.equals(ir.null()):
            raise TypeError('Null is not supported, default '
                            'value must be explicitly defined!')
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


def _table_array_view(translator, expr):
    ctx = translator.context
    table = expr.op().table
    query = ctx.get_compiled_expr(table)
    return '(\n{0}\n)'.format(util.indent(query, ctx.indent))


# ---------------------------------------------------------------------
# Timestamp arithmetic and other functions


def _timestamp_from_unix(translator, expr):
    op = expr.op()
    val, unit = op.args

    if unit == 'ms':
        raise ValueError('`ms` unit is not supported!')
    elif unit == 'us':
        raise ValueError('`us` unit is not supported!')

    arg = translator.translate(val)
    return 'toUInt32({0})'.format(arg)


def _timestamp_delta(translator, expr):
    op = expr.op()
    arg, offset = op.args
    formatted_arg = translator.translate(arg)
    return _timestamp_format_offset(offset, formatted_arg)


_clickhouse_delta_functions = {
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
    f = _clickhouse_delta_functions[type(offset)]
    return '{0}({1}, {2})'.format(f, arg, offset.n)


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


# _subtract_one = '({0} - 1)'.format
# _expr_transforms = {
#     ops.RowNumber: _subtract_one,
#     ops.DenseRank: _subtract_one,
#     ops.MinRank: _subtract_one,
#     ops.NTile: _subtract_one,
# }


# TODO: clickhouse uses differenct string functions
#       for ascii and utf-8 encodings,

_binary_infix_ops = {
    # Binary operations
    ops.Add: binary_infix_op('+'),
    ops.Subtract: binary_infix_op('-'),
    ops.Multiply: binary_infix_op('*'),
    ops.Divide: binary_infix_op('/'),
    ops.Power: fixed_arity('pow', 2),
    ops.Modulus: binary_infix_op('%'),

    # Comparisons
    ops.Equals: binary_infix_op('='),
    ops.NotEquals: binary_infix_op('!='),
    ops.GreaterEqual: binary_infix_op('>='),
    ops.Greater: binary_infix_op('>'),
    ops.LessEqual: binary_infix_op('<='),
    ops.Less: binary_infix_op('<'),

    # Boolean comparisons
    ops.And: binary_infix_op('AND'),
    ops.Or: binary_infix_op('OR'),
    ops.Xor: _xor,
}

_unary_ops = {
    ops.Negate: _negate,
    ops.Not: _not
}


_operation_registry = {
    # Unary operations
    ops.TypeOf: unary('toTypeName'),

    ops.Abs: unary('abs'),
    ops.Ceil: unary('ceil'),
    ops.Floor: unary('floor'),
    ops.Exp: unary('exp'),
    ops.Round: _round,

    ops.Sign: _sign,
    ops.Sqrt: unary('sqrt'),

    ops.Hash: _hash,

    ops.Log: _log,
    ops.Ln: unary('log'),
    ops.Log2: unary('log2'),
    ops.Log10: unary('log10'),

    # Unary aggregates
    ops.CMSMedian: agg('median'),
    # TODO: there is also a `uniq` function which is the
    #       recommended way to approximate cardinality
    ops.HLLCardinality: agg('uniqHLL12'),
    ops.Mean: agg('avg'),
    ops.Sum: agg('sum'),
    ops.Max: agg('max'),
    ops.Min: agg('min'),

    ops.StandardDev: agg_variance_like('stddev'),
    ops.Variance: agg_variance_like('var'),

    # ops.GroupConcat: fixed_arity('group_concat', 2),

    ops.Count: agg('count'),
    ops.CountDistinct: agg('uniq'),

    # string operations
    ops.StringLength: unary('length'),
    ops.Lowercase: unary('lower'),
    ops.Uppercase: unary('upper'),
    ops.Reverse: unary('reverse'),
    ops.Substring: _substring,
    ops.StringFind: _string_find,
    ops.FindInSet: _index_of,
    ops.StringReplace: fixed_arity('replaceAll', 3),

    # TODO: there are no concat_ws in clickhouse
    # ops.StringJoin: varargs('concat'),

    ops.StringSQLLike: binary_infix_op('LIKE'),
    ops.RegexSearch: fixed_arity('match', 2),
    # TODO: extractAll(haystack, pattern)[index + 1]
    ops.RegexExtract: _regex_extract,
    ops.RegexReplace: fixed_arity('replaceRegexpAll', 3),
    ops.ParseURL: _parse_url,

    # Timestamp operations
    ops.TimestampNow: lambda *args: 'now()',
    ops.ExtractYear: unary('toYear'),
    ops.ExtractMonth: unary('toMonth'),
    ops.ExtractDay: unary('toDayOfMonth'),
    ops.ExtractHour: unary('toHour'),
    ops.ExtractMinute: unary('toMinute'),
    ops.ExtractSecond: unary('toSecond'),

    # Other operations
    ops.E: lambda *args: 'e()',

    ir.Literal: literal,
    ir.ValueList: _value_list,

    ops.Cast: _cast,

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

    ops.TimestampDelta: _timestamp_delta,
    ops.TimestampFromUNIX: _timestamp_from_unix,

    transforms.ExistsSubquery: _exists_subquery,
    transforms.NotExistsSubquery: _exists_subquery,

    # RowNumber, and rank functions starts with 0 in Ibis-land
    # ops.RowNumber: lambda *args: 'row_number()',
    # ops.DenseRank: lambda *args: 'dense_rank()',
    # ops.MinRank: lambda *args: 'rank()',
    # ops.PercentRank: lambda *args: 'percent_rank()',

    # ops.FirstValue: unary('first_value'),
    # ops.LastValue: unary('last_value'),
    # ops.NthValue: _nth_value,
    # ops.Lag: _shift_like('lag'),
    # ops.Lead: _shift_like('lead'),
    # ops.NTile: _ntile,
}


def raise_error(translator, expr, *args):
    msg = 'Clickhouse backend doesn\'t support {0} operation!'
    op = expr.op()
    raise com.TranslationError(msg.format(type(op)))


def _null_literal(translator, expr):
    return 'NULL'


_undocumented_operations = {
    ir.NullLiteral: _null_literal,  # undocumented
    ops.IsNull: unary('isNull'),
    ops.NotNull: unary('isNotNull'),
    ops.IfNull: fixed_arity('ifNull', 2),
    ops.NullIf: fixed_arity('nullIf', 2),
    ops.Coalesce: varargs('coalesce')
}


_unsupported_ops = [
    ops.Truncate,
    ops.WindowOp,
    ops.DecimalPrecision,
    ops.DecimalScale,
    ops.ZeroIfNull,
    ops.NullIfZero,
    ops.BaseConvert,
    ops.CumulativeSum,
    ops.CumulativeMin,
    ops.CumulativeMax,
    ops.CumulativeMean,
    ops.CumulativeAny,
    ops.CumulativeAll,
    ops.IdenticalTo
]
_unsupported_ops = {k: raise_error for k in _unsupported_ops}

# TODO: toolz merge
_operation_registry.update(_undocumented_operations)
_operation_registry.update(_unsupported_ops)
_operation_registry.update(_unary_ops)
_operation_registry.update(_binary_infix_ops)
