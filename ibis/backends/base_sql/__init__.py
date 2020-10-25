"""
Shared functions for the SQL-based backends.

Eventually this should be converted to a base class inherited
from the SQL-based backends.
"""
import datetime
import itertools
import math

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.sql.compiler as comp
from ibis.backends.impala import identifiers


def _set_literal_format(translator, expr):
    value_type = expr.type().value_type

    formatted = [
        translator.translate(ir.literal(x, type=value_type))
        for x in expr.op().value
    ]

    return '(' + ', '.join(formatted) + ')'


def _boolean_literal_format(translator, expr):
    value = expr.op().value
    return 'TRUE' if value else 'FALSE'


def _string_literal_format(translator, expr):
    value = expr.op().value
    return "'{}'".format(value.replace("'", "\\'"))


def _number_literal_format(translator, expr):
    value = expr.op().value

    if math.isfinite(value):
        formatted = repr(value)
    else:
        if math.isnan(value):
            formatted_val = 'NaN'
        elif math.isinf(value):
            if value > 0:
                formatted_val = 'Infinity'
            else:
                formatted_val = '-Infinity'
        formatted = "CAST({!r} AS DOUBLE)".format(formatted_val)

    return formatted


def _interval_literal_format(translator, expr):
    return 'INTERVAL {} {}'.format(
        expr.op().value, expr.type().resolution.upper()
    )


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


literal_formatters = {
    'boolean': _boolean_literal_format,
    'number': _number_literal_format,
    'string': _string_literal_format,
    'interval': _interval_literal_format,
    'timestamp': _timestamp_literal_format,
    'date': _date_literal_format,
    'set': _set_literal_format,
}


def literal(translator, expr):
    """Return the expression as its literal value."""
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

    return literal_formatters[typeclass](translator, expr)


def quote_identifier(name, quotechar='`', force=False):
    """Add quotes to the `name` identifier if needed."""
    if force or name.count(' ') or name in identifiers.impala_identifiers:
        return '{0}{1}{0}'.format(quotechar, name)
    else:
        return name


# TODO move the name method to comp.ExprTranslator and use that instead
class BaseExprTranslator(comp.ExprTranslator):
    """Base expression translator."""

    @staticmethod
    def _name_expr(formatted_expr, quoted_name):
        return '{} AS {}'.format(formatted_expr, quoted_name)

    def name(self, translated, name, force=True):
        """Return expression with its identifier."""
        return self._name_expr(translated, quote_identifier(name, force=force))


parenthesize = '({})'.format


def format_call(translator, func, *args):
    formatted_args = []
    for arg in args:
        fmt_arg = translator.translate(arg)
        formatted_args.append(fmt_arg)

    return '{}({})'.format(func, ', '.join(formatted_args))


def fixed_arity(func_name, arity):
    def formatter(translator, expr):
        op = expr.op()
        if arity != len(op.args):
            raise com.IbisError('incorrect number of args')
        return format_call(translator, func_name, *op.args)

    return formatter


def needs_parens(op):
    if isinstance(op, ir.Expr):
        op = op.op()
    op_klass = type(op)
    # function calls don't need parens
    return op_klass in binary_infix_ops or op_klass in {
        ops.Negate,
        ops.IsNull,
        ops.NotNull,
    }


def binary_infix_op(infix_sym):
    def formatter(translator, expr):
        op = expr.op()

        left, right = op.args

        left_arg = translator.translate(left)
        right_arg = translator.translate(right)
        if needs_parens(left):
            left_arg = parenthesize(left_arg)

        if needs_parens(right):
            right_arg = parenthesize(right_arg)

        return '{} {} {}'.format(left_arg, infix_sym, right_arg)

    return formatter


def identical_to(translator, expr):
    op = expr.op()
    if op.args[0].equals(op.args[1]):
        return 'TRUE'

    left_expr = op.left
    right_expr = op.right
    left = translator.translate(left_expr)
    right = translator.translate(right_expr)

    if needs_parens(left_expr):
        left = parenthesize(left)
    if needs_parens(right_expr):
        right = parenthesize(right)
    return '{} IS NOT DISTINCT FROM {}'.format(left, right)


def xor(translator, expr):
    op = expr.op()

    left_arg = translator.translate(op.left)
    right_arg = translator.translate(op.right)

    if needs_parens(op.left):
        left_arg = parenthesize(left_arg)

    if needs_parens(op.right):
        right_arg = parenthesize(right_arg)

    return '({0} OR {1}) AND NOT ({0} AND {1})'.format(left_arg, right_arg)


def unary(func_name):
    return fixed_arity(func_name, 1)


def ifnull_workaround(translator, expr):
    op = expr.op()
    a, b = op.args

    # work around per #345, #360
    if isinstance(a, ir.DecimalValue) and isinstance(b, ir.IntegerValue):
        b = b.cast(a.type())

    return format_call(translator, 'isnull', a, b)


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


def _not(translator, expr):
    (arg,) = expr.op().args
    formatted_arg = translator.translate(arg)
    if needs_parens(arg):
        formatted_arg = parenthesize(formatted_arg)
    return 'NOT {}'.format(formatted_arg)


def not_null(translator, expr):
    formatted_arg = translator.translate(expr.op().args[0])
    return '{} IS NOT NULL'.format(formatted_arg)


def is_null(translator, expr):
    formatted_arg = translator.translate(expr.op().args[0])
    return '{} IS NULL'.format(formatted_arg)


def negate(translator, expr):
    arg = expr.op().args[0]
    formatted_arg = translator.translate(arg)
    if isinstance(expr, ir.BooleanValue):
        return _not(translator, expr)
    else:
        if needs_parens(arg):
            formatted_arg = parenthesize(formatted_arg)
        return '-{}'.format(formatted_arg)


def round(translator, expr):
    op = expr.op()
    arg, digits = op.args

    arg_formatted = translator.translate(arg)

    if digits is not None:
        digits_formatted = translator.translate(digits)
        return 'round({}, {})'.format(arg_formatted, digits_formatted)
    return 'round({})'.format(arg_formatted)


def sign(translator, expr):
    (arg,) = expr.op().args
    translated_arg = translator.translate(arg)
    translated_type = type_to_sql_string(expr.type())
    if expr.type() != dt.float:
        return 'CAST(sign({}) AS {})'.format(translated_arg, translated_type)
    return 'sign({})'.format(translated_arg)


def hash(translator, expr):
    op = expr.op()
    arg, how = op.args

    arg_formatted = translator.translate(arg)

    if how == 'fnv':
        return 'fnv_hash({})'.format(arg_formatted)
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


def reduction_format(translator, func_name, where, arg, *args):
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
        return reduction_format(translator, func_name, where, *args)

    return formatter


def variance_like(func_name):
    func_names = {
        'sample': '{}_samp'.format(func_name),
        'pop': '{}_pop'.format(func_name),
    }

    def formatter(translator, expr):
        arg, how, where = expr.op().args
        return reduction_format(translator, func_names[how], where, arg)

    return formatter


def count_distinct(translator, expr):
    arg, where = expr.op().args

    if where is not None:
        arg_formatted = translator.translate(where.ifelse(arg, None))
    else:
        arg_formatted = translator.translate(arg)
    return 'count(DISTINCT {})'.format(arg_formatted)


# ---------------------------------------------------------------------
# Scalar and array expression formatting

sql_type_names = {
    'int8': 'tinyint',
    'int16': 'smallint',
    'int32': 'int',
    'int64': 'bigint',
    'float': 'float',
    'float32': 'float',
    'double': 'double',
    'float64': 'double',
    'string': 'string',
    'boolean': 'boolean',
    'timestamp': 'timestamp',
    'decimal': 'decimal',
}


def type_to_sql_string(tval):
    if isinstance(tval, dt.Decimal):
        return 'decimal({}, {})'.format(tval.precision, tval.scale)
    name = tval.name.lower()
    try:
        return sql_type_names[name]
    except KeyError:
        raise com.UnsupportedBackendType(name)


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


def _string_join(translator, expr):
    op = expr.op()
    arg, strings = op.args
    return format_call(translator, 'concat_ws', arg, *strings)


def _string_like(translator, expr):
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
    op = expr.op()
    arg, unit = op.args

    arg_formatted = translator.translate(arg)
    try:
        unit = _base_unit_names[unit]
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


_base_unit_names = {
    'Y': 'Y',
    'Q': 'Q',
    'M': 'MONTH',
    'W': 'W',
    'D': 'J',
    'h': 'HH',
    'm': 'MI',
}


operation_registry = {
    # Unary operations
    ops.NotNull: not_null,
    ops.IsNull: is_null,
    ops.Negate: negate,
    ops.Not: _not,
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
    ops.StringJoin: _string_join,
    ops.StringSQLLike: _string_like,
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
}
