import itertools

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base_sqlalchemy import transforms

from . import case, helpers, literal, window


def binary_infix_op(infix_sym):
    def formatter(translator, expr):
        op = expr.op()

        left, right = op.args

        left_arg = translator.translate(left)
        right_arg = translator.translate(right)
        if helpers.needs_parens(left):
            left_arg = helpers.parenthesize(left_arg)

        if helpers.needs_parens(right):
            right_arg = helpers.parenthesize(right_arg)

        return '{} {} {}'.format(left_arg, infix_sym, right_arg)

    return formatter


def fixed_arity(func_name, arity):
    def formatter(translator, expr):
        op = expr.op()
        if arity != len(op.args):
            raise com.IbisError('incorrect number of args')
        return helpers.format_call(translator, func_name, *op.args)

    return formatter


def identical_to(translator, expr):
    op = expr.op()
    if op.args[0].equals(op.args[1]):
        return 'TRUE'

    left_expr = op.left
    right_expr = op.right
    left = translator.translate(left_expr)
    right = translator.translate(right_expr)

    if helpers.needs_parens(left_expr):
        left = helpers.parenthesize(left)
    if helpers.needs_parens(right_expr):
        right = helpers.parenthesize(right)
    return '{} IS NOT DISTINCT FROM {}'.format(left, right)


def xor(translator, expr):
    op = expr.op()

    left_arg = translator.translate(op.left)
    right_arg = translator.translate(op.right)

    if helpers.needs_parens(op.left):
        left_arg = helpers.parenthesize(left_arg)

    if helpers.needs_parens(op.right):
        right_arg = helpers.parenthesize(right_arg)

    return '({0} OR {1}) AND NOT ({0} AND {1})'.format(left_arg, right_arg)


def unary(func_name):
    return fixed_arity(func_name, 1)


def _reduction_format(translator, func_name, where, arg, *args):
    if where is not None:
        arg = where.ifelse(arg, ibis.NA)

    return '{}({})'.format(
        func_name,
        ', '.join(map(translator.translate, itertools.chain([arg], args))),
    )


def reduction(func_name):
    def formatter(translator, expr):
        op = expr.op()
        *args, where = op.args
        return _reduction_format(translator, func_name, where, *args)

    return formatter


def variance_like(func_name):
    func_names = {
        'sample': '{}_samp'.format(func_name),
        'pop': '{}_pop'.format(func_name),
    }

    def formatter(translator, expr):
        arg, how, where = expr.op().args
        return _reduction_format(translator, func_names[how], where, arg)

    return formatter


def not_null(translator, expr):
    formatted_arg = translator.translate(expr.op().args[0])
    return '{} IS NOT NULL'.format(formatted_arg)


def is_null(translator, expr):
    formatted_arg = translator.translate(expr.op().args[0])
    return '{} IS NULL'.format(formatted_arg)


def not_(translator, expr):
    (arg,) = expr.op().args
    formatted_arg = translator.translate(arg)
    if helpers.needs_parens(arg):
        formatted_arg = helpers.parenthesize(formatted_arg)
    return 'NOT {}'.format(formatted_arg)


def negate(translator, expr):
    arg = expr.op().args[0]
    formatted_arg = translator.translate(arg)
    if isinstance(expr, ir.BooleanValue):
        return not_(translator, expr)
    else:
        if helpers.needs_parens(arg):
            formatted_arg = helpers.parenthesize(formatted_arg)
        return '-{}'.format(formatted_arg)


def ifnull_workaround(translator, expr):
    op = expr.op()
    a, b = op.args

    # work around per #345, #360
    if isinstance(a, ir.DecimalValue) and isinstance(b, ir.IntegerValue):
        b = b.cast(a.type())

    return helpers.format_call(translator, 'isnull', a, b)


def sign(translator, expr):
    (arg,) = expr.op().args
    translated_arg = translator.translate(arg)
    translated_type = helpers.type_to_sql_string(expr.type())
    if expr.type() != dt.float:
        return 'CAST(sign({}) AS {})'.format(translated_arg, translated_type)
    return 'sign({})'.format(translated_arg)


def hashbytes(translator, expr):
    op = expr.op()
    arg, how = op.args

    arg_formatted = translator.translate(arg)

    if how == 'md5':
        return f'md5({arg_formatted})'
    elif how == 'sha1':
        return f'sha1({arg_formatted})'
    elif how == 'sha256':
        return f'sha256({arg_formatted})'
    elif how == 'sha512':
        return f'sha512({arg_formatted})'
    else:
        raise NotImplementedError(how)


def log(translator, expr):
    op = expr.op()
    arg, base = op.args
    arg_formatted = translator.translate(arg)

    if base is None:
        return 'ln({})'.format(arg_formatted)

    base_formatted = translator.translate(base)
    return 'log({}, {})'.format(base_formatted, arg_formatted)


def count_distinct(translator, expr):
    arg, where = expr.op().args

    if where is not None:
        arg_formatted = translator.translate(where.ifelse(arg, None))
    else:
        arg_formatted = translator.translate(arg)
    return 'count(DISTINCT {})'.format(arg_formatted)


def substring(translator, expr):
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


def string_find(translator, expr):
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


def find_in_set(translator, expr):
    op = expr.op()

    arg, str_list = op.args
    arg_formatted = translator.translate(arg)
    str_formatted = ','.join([x._arg.value for x in str_list])
    return "find_in_set({}, '{}') - 1".format(arg_formatted, str_formatted)


def string_join(translator, expr):
    op = expr.op()
    arg, strings = op.args
    return helpers.format_call(translator, 'concat_ws', arg, *strings)


def string_like(translator, expr):
    arg, pattern, _ = expr.op().args
    return '{} LIKE {}'.format(
        translator.translate(arg), translator.translate(pattern)
    )


def parse_url(translator, expr):
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


def extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.args[0])

        # This is pre-2.0 Impala-style, which did not used to support the
        # SQL-99 format extract($FIELD from expr)
        return "extract({}, '{}')".format(arg, sql_attr)

    return extract_field_formatter


def extract_epoch_seconds(t, expr):
    (arg,) = expr.op().args
    return 'unix_timestamp({})'.format(t.translate(arg))


def truncate(translator, expr):
    base_unit_names = {
        'Y': 'Y',
        'Q': 'Q',
        'M': 'MONTH',
        'W': 'W',
        'D': 'J',
        'h': 'HH',
        'm': 'MI',
    }
    op = expr.op()
    arg, unit = op.args

    arg_formatted = translator.translate(arg)
    try:
        unit = base_unit_names[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            '{!r} unit is not supported in timestamp truncate'.format(unit)
        )

    return "trunc({}, '{}')".format(arg_formatted, unit)


def interval_from_integer(translator, expr):
    # interval cannot be selected from impala
    op = expr.op()
    arg, unit = op.args
    arg_formatted = translator.translate(arg)

    return 'INTERVAL {} {}'.format(
        arg_formatted, expr.type().resolution.upper()
    )


def null_literal(translator, expr):
    return 'NULL'


def value_list(translator, expr):
    op = expr.op()
    formatted = [translator.translate(x) for x in op.values]
    return helpers.parenthesize(', '.join(formatted))


def cast(translator, expr):
    op = expr.op()
    arg, target_type = op.args
    arg_formatted = translator.translate(arg)

    if isinstance(arg, ir.CategoryValue) and target_type == dt.int32:
        return arg_formatted
    if isinstance(arg, ir.TemporalValue) and target_type == dt.int64:
        return '1000000 * unix_timestamp({})'.format(arg_formatted)
    else:
        sql_type = helpers.type_to_sql_string(target_type)
        return 'CAST({} AS {})'.format(arg_formatted, sql_type)


def varargs(func_name):
    def varargs_formatter(translator, expr):
        op = expr.op()
        return helpers.format_call(translator, func_name, *op.arg)

    return varargs_formatter


def between(translator, expr):
    op = expr.op()
    comp, lower, upper = [translator.translate(x) for x in op.args]
    return '{} BETWEEN {} AND {}'.format(comp, lower, upper)


def table_array_view(translator, expr):
    ctx = translator.context
    table = expr.op().table
    query = ctx.get_compiled_expr(table)
    return '(\n{}\n)'.format(util.indent(query, ctx.indent))


def table_column(translator, expr):
    op = expr.op()
    field_name = op.name
    quoted_name = helpers.quote_identifier(field_name, force=True)

    table = op.table
    ctx = translator.context

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if translator.permit_subquery and ctx.is_foreign_expr(table):
        proj_expr = table.projection([field_name]).to_array()
        return table_array_view(translator, proj_expr)

    if ctx.need_aliases():
        alias = ctx.get_ref(table)
        if alias is not None:
            quoted_name = '{}.{}'.format(alias, quoted_name)

    return quoted_name


def timestamp_op(func):
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


def timestamp_diff(translator, expr):
    op = expr.op()
    left, right = op.args

    return 'unix_timestamp({}) - unix_timestamp({})'.format(
        translator.translate(left), translator.translate(right)
    )


def _from_unixtime(translator, expr):
    arg = translator.translate(expr)
    return 'from_unixtime({}, "yyyy-MM-dd HH:mm:ss")'.format(arg)


def timestamp_from_unix(translator, expr):
    op = expr.op()

    val, unit = op.args
    val = util.convert_unit(val, unit, 's').cast('int32')

    arg = _from_unixtime(translator, val)
    return 'CAST({} AS timestamp)'.format(arg)


def exists_subquery(translator, expr):
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


def nth_value(translator, expr):
    op = expr.op()
    arg, rank = op.args

    arg_formatted = translator.translate(arg)
    rank_formatted = translator.translate(rank - 1)

    return 'first_value(lag({}, {}))'.format(arg_formatted, rank_formatted)


def shift_like(name):
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


def ntile(translator, expr):
    op = expr.op()
    arg, buckets = map(translator.translate, op.args)
    return 'ntile({})'.format(buckets)


def day_of_week_index(t, expr):
    (arg,) = expr.op().args
    return 'pmod(dayofweek({}) - 2, 7)'.format(t.translate(arg))


def day_of_week_name(t, expr):
    (arg,) = expr.op().args
    return 'dayname({})'.format(t.translate(arg))


binary_infix_ops = {
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
    ops.IdenticalTo: identical_to,
    # Boolean comparisons
    ops.And: binary_infix_op('AND'),
    ops.Or: binary_infix_op('OR'),
    ops.Xor: xor,
}


operation_registry = {
    # Unary operations
    ops.NotNull: not_null,
    ops.IsNull: is_null,
    ops.Negate: negate,
    ops.Not: not_,
    ops.IsNan: unary('is_nan'),
    ops.IsInf: unary('is_inf'),
    ops.IfNull: ifnull_workaround,
    ops.NullIf: fixed_arity('nullif', 2),
    ops.ZeroIfNull: unary('zeroifnull'),
    ops.NullIfZero: unary('nullifzero'),
    ops.Abs: unary('abs'),
    ops.BaseConvert: fixed_arity('conv', 3),
    ops.Ceil: unary('ceil'),
    ops.Floor: unary('floor'),
    ops.Exp: unary('exp'),
    ops.Round: round,
    ops.Sign: sign,
    ops.Sqrt: unary('sqrt'),
    ops.Hash: hash,
    ops.HashBytes: hashbytes,
    ops.Log: log,
    ops.Ln: unary('ln'),
    ops.Log2: unary('log2'),
    ops.Log10: unary('log10'),
    ops.DecimalPrecision: unary('precision'),
    ops.DecimalScale: unary('scale'),
    # Unary aggregates
    ops.CMSMedian: reduction('appx_median'),
    ops.HLLCardinality: reduction('ndv'),
    ops.Mean: reduction('avg'),
    ops.Sum: reduction('sum'),
    ops.Max: reduction('max'),
    ops.Min: reduction('min'),
    ops.StandardDev: variance_like('stddev'),
    ops.Variance: variance_like('var'),
    ops.GroupConcat: reduction('group_concat'),
    ops.Count: reduction('count'),
    ops.CountDistinct: count_distinct,
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
    ops.Substring: substring,
    ops.StrRight: fixed_arity('strright', 2),
    ops.Repeat: fixed_arity('repeat', 2),
    ops.StringFind: string_find,
    ops.Translate: fixed_arity('translate', 3),
    ops.FindInSet: find_in_set,
    ops.LPad: fixed_arity('lpad', 3),
    ops.RPad: fixed_arity('rpad', 3),
    ops.StringJoin: string_join,
    ops.StringSQLLike: string_like,
    ops.RegexSearch: fixed_arity('regexp_like', 2),
    ops.RegexExtract: fixed_arity('regexp_extract', 3),
    ops.RegexReplace: fixed_arity('regexp_replace', 3),
    ops.ParseURL: parse_url,
    # Timestamp operations
    ops.Date: unary('to_date'),
    ops.TimestampNow: lambda *args: 'now()',
    ops.ExtractYear: extract_field('year'),
    ops.ExtractMonth: extract_field('month'),
    ops.ExtractDay: extract_field('day'),
    ops.ExtractQuarter: extract_field('quarter'),
    ops.ExtractEpochSeconds: extract_epoch_seconds,
    ops.ExtractWeekOfYear: fixed_arity('weekofyear', 1),
    ops.ExtractHour: extract_field('hour'),
    ops.ExtractMinute: extract_field('minute'),
    ops.ExtractSecond: extract_field('second'),
    ops.ExtractMillisecond: extract_field('millisecond'),
    ops.TimestampTruncate: truncate,
    ops.DateTruncate: truncate,
    ops.IntervalFromInteger: interval_from_integer,
    # Other operations
    ops.E: lambda *args: 'e()',
    ops.Literal: literal.literal,
    ops.NullLiteral: null_literal,
    ops.ValueList: value_list,
    ops.Cast: cast,
    ops.Coalesce: varargs('coalesce'),
    ops.Greatest: varargs('greatest'),
    ops.Least: varargs('least'),
    ops.Where: fixed_arity('if', 3),
    ops.Between: between,
    ops.Contains: binary_infix_op('IN'),
    ops.NotContains: binary_infix_op('NOT IN'),
    ops.SimpleCase: case.simple_case,
    ops.SearchedCase: case.searched_case,
    ops.TableColumn: table_column,
    ops.TableArrayView: table_array_view,
    ops.DateAdd: timestamp_op('date_add'),
    ops.DateSub: timestamp_op('date_sub'),
    ops.DateDiff: timestamp_op('datediff'),
    ops.TimestampAdd: timestamp_op('date_add'),
    ops.TimestampSub: timestamp_op('date_sub'),
    ops.TimestampDiff: timestamp_diff,
    ops.TimestampFromUNIX: timestamp_from_unix,
    transforms.ExistsSubquery: exists_subquery,
    transforms.NotExistsSubquery: exists_subquery,
    # RowNumber, and rank functions starts with 0 in Ibis-land
    ops.RowNumber: lambda *args: 'row_number()',
    ops.DenseRank: lambda *args: 'dense_rank()',
    ops.MinRank: lambda *args: 'rank()',
    ops.PercentRank: lambda *args: 'percent_rank()',
    ops.FirstValue: unary('first_value'),
    ops.LastValue: unary('last_value'),
    ops.NthValue: nth_value,
    ops.Lag: shift_like('lag'),
    ops.Lead: shift_like('lead'),
    ops.WindowOp: window.window,
    ops.NTile: ntile,
    ops.DayOfWeekIndex: day_of_week_index,
    ops.DayOfWeekName: day_of_week_name,
    **binary_infix_ops,
}
