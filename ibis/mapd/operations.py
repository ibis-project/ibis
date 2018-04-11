from six import StringIO
from datetime import date, datetime
from ibis.mapd.identifiers import quote_identifier

import ibis.common as com
import ibis.util as util
import ibis.expr.rules as rlz
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.sql.transforms as transforms

from ibis.expr.types import NumericValue


def _cast(translator, expr):
    from ibis.mapd.client import MapDDataType

    op = expr.op()
    arg, target = op.args
    arg_ = translator.translate(arg)
    type_ = str(MapDDataType.from_ibis(target, nullable=False))

    return 'CAST({0!s} AS {1!s})'.format(arg_, type_)


def _between(translator, expr):
    op = expr.op()
    arg_, lower_, upper_ = map(translator.translate, op.args)
    return '{0!s} BETWEEN {1!s} AND {2!s}'.format(arg_, lower_, upper_)


def _negate(translator, expr):
    arg = expr.op().args[0]
    if isinstance(expr, ir.BooleanValue):
        arg_ = translator.translate(arg)
        return 'NOT {0!s}'.format(arg_)
    else:
        arg_ = _parenthesize(translator, arg)
        return '-{0!s}'.format(arg_)


def _not(translator, expr):
    return 'NOT {}'.format(*map(translator.translate, expr.op().args))


def _parenthesize(translator, expr):
    op = expr.op()
    op_klass = type(op)

    # function calls don't need parens
    what_ = translator.translate(expr)
    if (op_klass in _binary_infix_ops) or (op_klass in _unary_ops):
        return '({0!s})'.format(what_)
    else:
        return what_


def fixed_arity(func_name, arity):
    def formatter(translator, expr):
        op = expr.op()
        arg_count = len(op.args)
        if arity != arg_count:
            msg = 'Incorrect number of args {0} instead of {1}'
            raise com.UnsupportedOperationError(msg.format(arg_count, arity))
        return _call(translator, func_name, *op.args)
    return formatter


def unary(func_name):
    return fixed_arity(func_name, 1)


def agg(func):
    def formatter(translator, expr):
        return _aggregate(translator, func, *expr.op().args)
    return formatter


def agg_variance_like(func):
    variants = {'sample': '{0}Samp'.format(func),
                'pop': '{0}Pop'.format(func)}

    def formatter(translator, expr):
        arg, how, where = expr.op().args
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
        return _call(translator, func_name, *op.arg)
    return varargs_formatter


def _substring(translator, expr):
    # arg_ is the formatted notation
    op = expr.op()
    arg, start, length = op.args
    arg_, start_ = translator.translate(arg), translator.translate(start)

    # MapD is 1-indexed
    if length is None or isinstance(length.op(), ops.Literal):
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
        raise com.UnsupportedOperationError(
            "String find doesn't support start argument"
        )

    return _call(translator, 'position', arg, substr) + ' - 1'


def _regex_extract(translator, expr):
    op = expr.op()
    arg, pattern, index = op.args
    arg_, pattern_ = translator.translate(arg), translator.translate(pattern)

    if index is not None:
        index_ = translator.translate(index)
        return 'extractAll({0}, {1})[{2} + 1]'.format(arg_, pattern_, index_)

    return 'extractAll({0}, {1})'.format(arg_, pattern_)


def _parse_url(translator, expr):
    op = expr.op()
    arg, extract, key = op.args

    if extract == 'HOST':
        return _call(translator, 'domain', arg)
    elif extract == 'PROTOCOL':
        return _call(translator, 'protocol', arg)
    elif extract == 'PATH':
        return _call(translator, 'path', arg)
    elif extract == 'QUERY':
        if key is not None:
            return _call(translator, 'extractURLParameter', arg, key)
        else:
            return _call(translator, 'queryString', arg)
    else:
        raise com.UnsupportedOperationError(
            'Parse url with extract {0} is not supported'.format(extract)
        )


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

    if digits is not None:
        return _call(translator, 'round', arg, digits)
    else:
        return _call(translator, 'round', arg)


def _value_list(translator, expr):
    op = expr.op()
    values_ = map(translator.translate, op.values)
    return '({0})'.format(', '.join(values_))


def _interval_format(translator, expr):
    dtype = expr.type()
    if dtype.unit in {'ms', 'us', 'ns'}:
        raise com.UnsupportedOperationError(
            "MapD doesn't support subsecond interval resolutions")

    return 'INTERVAL {} {}'.format(expr.op().value, dtype.resolution.upper())


def _interval_from_integer(translator, expr):
    op = expr.op()
    arg, unit = op.args

    dtype = expr.type()
    if dtype.unit in {'ms', 'us', 'ns'}:
        raise com.UnsupportedOperationError(
            "MapD doesn't support subsecond interval resolutions")

    arg_ = translator.translate(arg)
    return 'INTERVAL {} {}'.format(arg_, dtype.resolution.upper())


def literal(translator, expr):
    value = expr.op().value
    if isinstance(expr, ir.BooleanValue):
        return '1' if value else '0'
    elif isinstance(expr, ir.StringValue):
        return "'{0!s}'".format(value.replace("'", "\\'"))
    elif isinstance(expr, ir.NumericValue):
        return repr(value)
    elif isinstance(expr, ir.IntervalValue):
        return _interval_format(translator, expr)
    elif isinstance(expr, ir.TimestampValue):
        if isinstance(value, datetime):
            if value.microsecond != 0:
                msg = 'Unsupported subsecond accuracy {}'
                raise ValueError(msg.format(value))
            value = value.strftime('%Y-%m-%d %H:%M:%S')
        return "toDateTime('{0!s}')".format(value)
    elif isinstance(expr, ir.DateValue):
        if isinstance(value, date):
            value = value.strftime('%Y-%m-%d')
        return "toDate('{0!s}')".format(value)
    elif isinstance(expr, ir.ArrayValue):
        return str(list(value))
    else:
        raise NotImplementedError(type(expr))


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
        """

        :return:
        """
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
    return '(\n{0}\n)'.format(util.indent(query, ctx.indent))


def _timestamp_from_unix(translator, expr):
    op = expr.op()
    arg, unit = op.args

    if unit in {'ms', 'us', 'ns'}:
        raise ValueError('`{}` unit is not supported!'.format(unit))

    return _call(translator, 'toDateTime', arg)


def _truncate(translator, expr):
    op = expr.op()
    arg, unit = op.args

    converters = {
        'Y': 'toStartOfYear',
        'M': 'toStartOfMonth',
        'W': 'toMonday',
        'D': 'toDate',
        'h': 'toStartOfHour',
        'm': 'toStartOfMinute',
        's': 'toDateTime'
    }

    try:
        converter = converters[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            'Unsupported truncate unit {}'.format(unit)
        )

    return _call(translator, converter, arg)


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

    # TODO(kszucs): table aliasing is partially supported
    # if ctx.need_aliases():
    #     alias = ctx.get_ref(table)
    #     if alias is not None:
    #         quoted_name = '{0}.{1}'.format(alias, quoted_name)

    return quoted_name


def _string_split(translator, expr):
    value, sep = expr.op().args
    return 'splitByString({}, {})'.format(
        translator.translate(sep),
        translator.translate(value)
    )


def _string_join(translator, expr):
    sep, elements = expr.op().args
    assert isinstance(elements.op(), ops.ValueList), \
        'elements must be a ValueList, got {}'.format(type(elements.op()))
    return 'arrayStringConcat([{}], {})'.format(
        ', '.join(map(translator.translate, elements)),
        translator.translate(sep),
    )


def _string_repeat(translator, expr):
    value, times = expr.op().args
    result = 'arrayStringConcat(arrayMap(x -> {}, range({})))'.format(
        translator.translate(value), translator.translate(times)
    )
    return result


def _string_like(translator, expr):
    value, pattern = expr.op().args[:2]
    return '{} LIKE {}'.format(
        translator.translate(value), translator.translate(pattern)
    )


def raise_error(translator, expr, *args):
    msg = "MapD backend doesn't support {0} operation!"
    op = expr.op()
    raise com.UnsupportedOperationError(msg.format(type(op)))


def _null_literal(translator, expr):
    return 'Null'


def _null_if_zero(translator, expr):
    op = expr.op()
    arg = op.args[0]
    arg_ = translator.translate(arg)
    return 'nullIf({0}, 0)'.format(arg_)


def _zero_if_null(translator, expr):
    op = expr.op()
    arg = op.args[0]
    arg_ = translator.translate(arg)
    return 'ifNull({0}, 0)'.format(arg_)


# AGGREGATION

class ApproxCountDistinct(ops.Reduction):
    """
    Returns the approximate count of distinct values of x with defined
    expected error rate e
    """
    arg_x = ops.Arg(rlz.column(rlz.numeric))
    arg_e = ops.Arg(rlz.column(rlz.numeric))
    where = ops.Arg(rlz.boolean, default=None)

    def output_type(self):
        return ops.dt.float64.scalar_type()


# MATH

class Degrees(ops.UnaryOp):
    """Converts radians to degrees"""
    arg = ops.Arg(rlz.floating)
    output_type = rlz.shape_like('arg', ops.dt.float)


class PI(ops.Constant):
    """Converts radians to degrees"""
    def output_type(self):
        return ops.dt.float64.scalar_type()


class Radians(ops.UnaryOp):
    """Converts radians to degrees"""
    arg = ops.Arg(rlz.floating)
    output_type = rlz.shape_like('arg', ops.dt.float)


class Truncate(ops.NumericBinaryOp):
    """Truncates x to y decimal places"""


# TRIGONOMETRY

class TrigonometryUnary(ops.UnaryOp):
    """Trigonometry base unary"""
    def output_type(self):
        return ops.dt.float64.scalar_type()


class TrigonometryBinary(ops.BinaryOp):
    """Trigonometry base binary"""


class Acos(TrigonometryUnary):
    """Returns the arc cosine of x"""


class Asin(TrigonometryUnary):
    """Returns the arc sine of x"""


class Atan(TrigonometryUnary):
    """Returns the arc tangent of x"""


class Atan2(TrigonometryBinary):
    """Returns the arc tangent of x and y"""


class Cos(TrigonometryUnary):
    """Returns the cosine of x"""


class Cot(TrigonometryUnary):
    """Returns the cotangent of x"""


class Sin(TrigonometryUnary):
    """Returns the sine of x"""


class Tan(TrigonometryUnary):
    """Returns the tangent of x"""


# GEOMETRIC

class DistanceInMeters(ops.ValueOp):
    """
    Calculates distance in meters between two WGS-84 positions.

    """
    fromLon = ops.Arg(rlz.column(rlz.numeric))
    fromLat = ops.Arg(rlz.column(rlz.numeric))
    toLon = ops.Arg(rlz.column(rlz.numeric))
    toLat = ops.Arg(rlz.column(rlz.numeric))
    output_type = rlz.shape_like('arg', ops.dt.float)


class Conv_4326_900913_X(ops.UnaryOp):
    """
    Converts WGS-84 latitude to WGS-84 Web Mercator x coordinate.
    """


class Conv_4326_900913_Y(ops.UnaryOp):
    """
    Converts WGS-84 longitude to WGS-84 Web Mercator y coordinate.

    """


# String

class StringLengthBytes(ops.UnaryOp):

    """
    Compute length in bytes of strings

    Returns
    -------
    length : int32
    """

    output_type = rlz.shape_like('arg', ops.dt.binary)


# https://www.mapd.com/docs/latest/mapd-core-guide/dml/
_binary_infix_ops = {
    # math
    ops.Add: binary_infix_op('+'),
    ops.Subtract: binary_infix_op('-'),
    ops.Multiply: binary_infix_op('*'),
    ops.Divide: binary_infix_op('/'),
    ops.Power: fixed_arity('power', 2),
    # comparison
    ops.Equals: binary_infix_op('='),
    ops.NotEquals: binary_infix_op('<>'),
    ops.GreaterEqual: binary_infix_op('>='),
    ops.Greater: binary_infix_op('>'),
    ops.LessEqual: binary_infix_op('<='),
    ops.Less: binary_infix_op('<'),
    # logical
    ops.And: binary_infix_op('AND'),
    ops.Or: binary_infix_op('OR'),
}

_unary_ops = {
    # logical
    ops.Negate: _negate,
    ops.Not: _not,
}

_comparison_ops = {
    ops.IsNull: unary('is null'),
    ops.Between: _between,
    ops.NullIf: fixed_arity('nullif', 2),
    ops.NotNull: unary('is not null'),
    ops.Contains: binary_infix_op('in'),
    ops.NotContains: binary_infix_op('not in'),
}


_math_ops = {
    ops.Abs: unary('abs'),
    ops.Ceil: unary('ceil'),
    Degrees: unary('degrees'),  # MapD function
    ops.Exp: unary('exp'),
    ops.Floor: unary('floor'),
    ops.Log: unary('log'),
    ops.Ln: unary('ln'),
    ops.Log10: unary('log10'),
    ops.Modulus: unary('mod'),
    PI: lambda *args: 'pi()',
    # ops.Power: binary('power'),  # TODO: check if it is necessary
    Radians: unary('radians'),
    ops.Round: _round,
    ops.Sign: _sign,
    ops.Sqrt: unary('sqrt'),
    Truncate: fixed_arity('truncate', 2)
}

_stats_ops = {
    # CORRELATION(x, y)	CORRELATION_FLOAT(x, y)	Alias of CORR. Returns the coefficient of correlation of a set of number pairs.
    # CORR(x, y)	CORR_FLOAT(x, y)	Returns the coefficient of correlation of a set of number pairs.
    # COVAR_POP(x, y)	COVAR_POP_FLOAT(x, y)	Returns the population covariance of a set of number pairs.
    # COVAR_SAMP(x, y)	COVAR_SAMP_FLOAT(x, y)	Returns the sample covariance of a set of number pairs.
    ops.StandardDev: agg_variance_like('stddev'),
    # STDDEV(x)	STDDEV_FLOAT(x)	Alias of STDDEV_SAMP. Returns sample standard deviation of the value.
    # STDDEV_POP(x)	STDDEV_POP_FLOAT(x)	Returns the population standard the standard deviation of the value.
    # STDDEV_SAMP(x)	STDDEV_SAMP_FLOAT(x)	Returns the sample standard deviation of the value.
    ops.Variance: agg_variance_like('var'),
    # VARIANCE(x)	VARIANCE_FLOAT(x)	Alias of VAR_SAMP. Returns the sample variance of the value.
    # VAR_POP(x)	VAR_POP_FLOAT(x)	Returns the population variance sample variance of the value.
    # VAR_SAMP(x)	VAR_SAMP_FLOAT(x)	Returns the sample variance of the value.
}


_trigonometric_ops = {
    Acos: unary('acos'),
    Asin: unary('asin'),
    Atan: unary('atan'),
    Atan2: fixed_arity('atan2', 2),
    Cos: unary('cos'),
    Cot: unary('cot'),
    Sin: unary('sin'),
    Tan: unary('tan')
}

_geometric_ops = {
    DistanceInMeters: fixed_arity('distance_in_meters', 4),
    Conv_4326_900913_X: unary('conv_4326_900913_x'),
    Conv_4326_900913_Y: unary('conv_4326_900913_y')
}

_string_ops = {
    ops.StringLength: unary('char_length'),
    StringLengthBytes: unary('length'),
    # ops.RegexSearch: fixed_arity('match', 2),
    # ops.RegexExtract: _regex_extract,
    # ops.RegexReplace: fixed_arity('replaceRegexpAll', 3),
    # str LIKE pattern	'ab' LIKE 'ab'	Returns true if the string matches the pattern
    # str NOT LIKE pattern	'ab' NOT LIKE 'cd'	Returns true if the string does not match the pattern
    # str ILIKE pattern	'AB' ILIKE 'ab'	Case-insensitive LIKE
    # str REGEXP POSIX pattern	'^[a-z]+r$'	Lowercase string ending with r
    # REGEXP_LIKE ( str , POSIX pattern )	'^[hc]at'	cat or hat
}

_date_ops = {
    ops.Date: unary('toDate'),
    ops.DateTruncate: _truncate,

    ops.TimestampNow: lambda *args: 'now()',
    ops.TimestampTruncate: _truncate,
    ops.TimeTruncate: _truncate,
    ops.IntervalFromInteger: _interval_from_integer,

    ops.ExtractYear: unary('toYear'),
    ops.ExtractMonth: unary('toMonth'),
    ops.ExtractDay: unary('toDayOfMonth'),
    ops.ExtractHour: unary('toHour'),
    ops.ExtractMinute: unary('toMinute'),
    ops.ExtractSecond: unary('toSecond'),

    ops.DateAdd: binary_infix_op('+'),
    ops.DateSub: binary_infix_op('-'),
    ops.DateDiff: binary_infix_op('-'),
    ops.TimestampAdd: binary_infix_op('+'),
    ops.TimestampSub: binary_infix_op('-'),
    ops.TimestampDiff: binary_infix_op('-'),
    ops.TimestampFromUNIX: _timestamp_from_unix,
}

_agg_ops = {
    # TODO: this function receive a x and e parameter
    ApproxCountDistinct: agg('approx_count_cistinct'),
    ops.Count: agg('count'),
    ops.CountDistinct: agg('count'),  # this function receive a x parameter
    ops.Mean: agg('avg'),
    ops.Max: agg('max'),
    ops.Min: agg('min'),
    ops.Sum: agg('sum'),
}

_general_ops = {
    # Unary operations
    ops.Literal: literal,
    ops.ValueList: _value_list,
    ops.Cast: _cast,
    ops.Where: fixed_arity('if', 3),
    ops.SimpleCase: _simple_case,
    ops.SearchedCase: _searched_case,
    ops.TableColumn: _table_column,
    ops.TableArrayView: _table_array_view,
    transforms.ExistsSubquery: _exists_subquery,
    transforms.NotExistsSubquery: _exists_subquery,
    ops.ArrayLength: unary('length'),
    ops.Coalesce: varargs('coalesce'),
}


# _unsupported_ops = []
# _unsupported_ops = {k: raise_error for k in _unsupported_ops}

_operation_registry = {}

_operation_registry.update(_general_ops)
_operation_registry.update(_binary_infix_ops)
_operation_registry.update(_unary_ops)
_operation_registry.update(_comparison_ops)
_operation_registry.update(_math_ops)
_operation_registry.update(_stats_ops)
_operation_registry.update(_trigonometric_ops)
_operation_registry.update(_geometric_ops)
_operation_registry.update(_string_ops)
_operation_registry.update(_date_ops)
_operation_registry.update(_agg_ops)
# _operation_registry.update(_unsupported_ops)


def sin(numeric_value):
    return Sin(numeric_value).to_expr()


NumericValue.sin = sin
