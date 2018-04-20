from datetime import date, datetime
from ibis.mapd.identifiers import quote_identifier
from six import StringIO

import ibis
import ibis.common as com
import ibis.util as util
import ibis.expr.rules as rlz
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.sql.transforms as transforms


def _is_floating(*args):
    for arg in args:
        if isinstance(arg, ir.FloatingColumn):
            return True
    return False


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
    formatter.__name__ = func_name
    return formatter


def unary(func_name):
    return fixed_arity(func_name, 1)


def _reduction_format(
    translator,
    func_name,
    sql_func_name=None,
    sql_signature='{}({})',
    arg=None, args=None, where=None
):
    if not sql_func_name:
        sql_func_name = func_name

    if where is not None:
        arg = where.ifelse(arg, ibis.NA)

    return sql_signature.format(
        sql_func_name,
        ', '.join(map(translator.translate, [arg] + list(args)))
    )


def _reduction(func_name, sql_func_name=None, sql_signature='{}({})'):
    def formatter(translator, expr):
        op = expr.op()

        # HACK: support trailing arguments
        where = op.where
        args = [arg for arg in op.args if arg is not where]

        return _reduction_format(
            translator, func_name, sql_func_name, sql_signature,
            args[0], args[1:], where
        )

    formatter.__name__ = func_name
    return formatter


def _variance_like(func):
    variants = {
        'sample': '{}_SAMP'.format(func),
        'pop': '{}_POP'.format(func)
    }

    def formatter(translator, expr):
        arg, how, where = expr.op().args

        func_type = '' if not _is_floating(arg) else '_FLOAT'

        return _reduction_format(
            translator, variants[how].upper() + func_type,
            None, '{}({})',
            arg, [], where
        )

    formatter.__name__ = func
    return formatter


def unary_prefix_op(prefix_op):
    def formatter(translator, expr):
        op = expr.op()
        arg = _parenthesize(translator, op.args[0])

        return '{0!s} {1!s}'.format(prefix_op.upper(), arg)

    formatter.__name__ = prefix_op
    return formatter


def binary_infix_op(infix_sym):
    def formatter(translator, expr):
        op = expr.op()

        left, right = op.args[0], op.args[1]
        left_ = _parenthesize(translator, left)
        right_ = _parenthesize(translator, right)

        return '{0!s} {1!s} {2!s}'.format(left_, infix_sym, right_)

    return formatter


def timestamp_binary_infix_op(func_name, infix_sym, timestamp_code):
    def formatter(translator, expr):
        op = expr.op()

        arg, unit = op.args[0], op.args[1]
        arg_ = _parenthesize(translator, arg)

        if unit.upper() in [u.upper() for u in timestamp_code.keys()]:
            converter = timestamp_code[unit].upper()
        elif unit.upper() in [u.upper() for u in _timestamp_units.values()]:
            converter = unit.upper()
        else:
            raise ValueError('`{}` unit is not supported!'.format(unit))

        return '{0!s}({1!s} {2!s} {3!s})'.format(
            func_name, converter, infix_sym, arg_
        )
    return formatter


def _call(translator, func, *args):
    args_ = ', '.join(map(translator.translate, args))
    return '{0!s}({1!s})'.format(func, args_)


def _call_date_trunc(translator, func, *args):
    args_ = ', '.join(map(translator.translate, args))
    return 'DATE_TRUNC({0!s}, {1!s})'.format(func, args_)


def _aggregate(translator, func, arg, where=None):
    if where is not None:
        return _call(translator, func + 'If', arg, where)
    else:
        return _call(translator, func, arg)


def _extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.args[0])
        return 'EXTRACT({} FROM {})'.format(sql_attr, arg)
    return extract_field_formatter


def _general_date_field_add(data_type_name):
    if data_type_name not in ('date', 'timestamp'):
        raise NotImplemented('{} not implemented.'.format(data_type_name))

    def __general_date_field_add(translator, expr):
        op = expr.op()
        self = translator.translate(op.args[0])
        value = translator.translate(op.args[1])
        return '{}({},{},{})'.format(data_type_name, self, value)

    return __general_date_field_add


def compile_corr(translator, expr):
    # pull out the arguments to the expression
    args = expr.op().args

    x, y, how, where = args

    f_type = '' if not _is_floating(x, y) else '_FLOAT'

    # compile the argument
    compiled_x = translator.translate(x)
    compiled_y = translator.translate(y)

    return 'CORR{}({}, {})'.format(f_type, compiled_x, compiled_y)


def compile_cov(translator, expr):
    # pull out the arguments to the expression
    args = expr.op().args

    x, y, how, where = args

    f_type = '' if not _is_floating(x, y) else '_FLOAT'

    # compile the argument
    compiled_x = translator.translate(x)
    compiled_y = translator.translate(y)

    return 'COVAR_{}{}({}, {})'.format(
        how[:4].upper(), f_type, compiled_x, compiled_y
    )


def compile_length(func_name='length', sql_func_name='CHAR_LENGTH'):
    def _compile_lenght(translator, expr):
        # pull out the arguments to the expression
        arg = expr.op().args[0]
        # compile the argument
        compiled_arg = translator.translate(arg)
        return '{}({})'.format(sql_func_name, compiled_arg)
    _compile_lenght.__name__ = func_name
    return _compile_lenght


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
    return 'SIGN({})'.format(translator.translate(arg))


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

    return '{1}, (sign){0}'.format(expr.op().value, dtype.resolution.upper())


def _interval_from_integer(translator, expr):
    op = expr.op()
    arg, unit = op.args

    dtype = expr.type()
    if dtype.unit in {'ms', 'us', 'ns'}:
        raise com.UnsupportedOperationError(
            "MapD doesn't support subsecond interval resolutions")

    arg_ = translator.translate(arg)
    return '{}, (sign){}'.format(dtype.resolution.upper(), arg_)


def _timestamp_op(func, op_sign='+'):
    def _formatter(translator, expr):
        op = expr.op()
        left, right = op.args

        formatted_left = translator.translate(left)
        formatted_right = translator.translate(right)

        if isinstance(left, ir.DateValue):
            formatted_left = 'CAST({} as timestamp)'.format(formatted_left)

        return '{}({}, {})'.format(
            func,
            formatted_right.replace('(sign)', op_sign),
            formatted_left
        )

    return _formatter


def _timestamp_diff(sql_func, timestamp_code):
    def _formatter(translator, expr):
        op = expr.op()
        left, right, unit = op.args

        formatted_left = translator.translate(left)
        formatted_right = translator.translate(right)

        formatted_unit = timestamp_code[unit]

        if isinstance(left, ir.DateValue):
            formatted_left = 'CAST({} as timestamp)'.format(formatted_left)

        if isinstance(right, ir.DateValue):
            formatted_right = 'CAST({} as timestamp)'.format(formatted_right)

        return '{}({}, {}, {})'.format(
            sql_func,
            formatted_unit,
            formatted_left,
            formatted_right
        )

    _formatter.__name__ = 'diff'
    return _formatter


def _set_literal_format(translator, expr):
    value_type = expr.type().value_type

    formatted = [translator.translate(ir.literal(x, type=value_type))
                 for x in expr.op().value]

    return '({})'.format(', '.join(formatted))


def literal(translator, expr):
    value = expr.op().value
    if isinstance(expr, ir.BooleanValue):
        return '1' if value else '0'
    elif isinstance(expr, ir.StringValue):
        return "'{0!s}'".format(value.replace("'", "\\'"))
    elif isinstance(expr, ir.NumericValue):
        return repr(value)
    elif isinstance(expr, ir.SetScalar):
        return _set_literal_format(translator, expr)
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


def date_truncate(translator, expr):
    op = expr.op()
    arg, unit = op.args

    timestamp_code = {
        'Y': 'YEAR',
        'M': 'MONTH',
        'D': 'DAY',
        'W': 'WEEK',
        'Q': 'QUARTER',
        'h': 'HOUR',
        'm': 'MINUTE',
        's': 'SECOND',
    }

    if unit.upper() in [u.upper() for u in timestamp_code.keys()]:
        converter = timestamp_code[unit].upper()
    elif unit.upper() in [u.upper() for u in _timestamp_units.values()]:
        converter = unit.upper()
    else:
        raise ValueError('`{}` unit is not supported!'.format(unit))

    return _call_date_trunc(translator, converter, arg)


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

class CountDistinct(ops.CountDistinct):
    """
    Returns the approximate count of distinct values of x with defined
    expected error rate e
    """
    approx = ops.Arg(rlz.integer)


# MATH
class Log(ops.Ln):
    """

    """


class Truncate(ops.NumericBinaryOp):
    """Truncates x to y decimal places"""
    output_type = rlz.shape_like('left', ops.dt.float)


# GEOMETRIC

class Distance_In_Meters(ops.ValueOp):
    """
    Calculates distance in meters between two WGS-84 positions.

    """
    fromLon = ops.Arg(rlz.column(rlz.numeric))
    fromLat = ops.Arg(rlz.column(rlz.numeric))
    toLon = ops.Arg(rlz.column(rlz.numeric))
    toLat = ops.Arg(rlz.column(rlz.numeric))
    output_type = rlz.shape_like('fromLon', ops.dt.float)


class Conv_4326_900913_X(ops.UnaryOp):
    """
    Converts WGS-84 latitude to WGS-84 Web Mercator x coordinate.
    """
    output_type = rlz.shape_like('arg', ops.dt.float)


class Conv_4326_900913_Y(ops.UnaryOp):
    """
    Converts WGS-84 longitude to WGS-84 Web Mercator y coordinate.

    """
    output_type = rlz.shape_like('arg', ops.dt.float)


# DATE/TIME OPERATIONS


_timestamp_units = dict(ops._date_units)
_timestamp_units.update(ops._time_units)
_timestamp_units.update(dict(
    millennium='MILLENNIUM',
    MILLENNIUM='MILLENNIUM',

    century='CENTURY',
    CENTURY='CENTURY',

    DECADE='DECADE',
    decade='decade',

    quarterday='quarterday',
    QUARTERDAY='QUARTERDAY',

    DOW='DOW',
    ISODOW='DOW',
    DOY='DOY',
    EPOCH='EPOCH'
))


class TimestampExtract(ops.TimestampUnaryOp):
    unit = ops.Arg(rlz.isin(_timestamp_units))
    output_type = rlz.shape_like('arg', ops.dt.int32)


class TimestampTruncate(ops.TimestampTruncate):
    unit = ops.Arg(rlz.isin(_timestamp_units))


class DateTruncate(ops.DateTruncate):
    unit = ops.Arg(rlz.isin(_timestamp_units))


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

# COMPARISON
_comparison_ops = {
    ops.IsNull: unary('is null'),
    ops.Between: _between,
    ops.NullIf: fixed_arity('nullif', 2),
    ops.NotNull: unary('is not null'),
    ops.Contains: binary_infix_op('in'),
    ops.NotContains: binary_infix_op('not in'),
}


# MATH
_math_ops = {
    ops.Abs: unary('abs'),
    ops.Ceil: unary('ceil'),
    ops.Degrees: unary('degrees'),  # MapD function
    ops.Exp: unary('exp'),
    ops.Floor: unary('floor'),
    Log: unary('log'),  # MapD Log wrap to IBIS Ln
    ops.Ln: unary('ln'),
    ops.Log10: unary('log10'),
    ops.Modulus: fixed_arity('mod', 2),
    ops.Pi: fixed_arity('pi', 0),
    ops.Radians: unary('radians'),
    ops.Round: _round,
    ops.Sign: _sign,
    ops.Sqrt: unary('sqrt'),
    Truncate: fixed_arity('truncate', 2)
}

# STATS
_stats_ops = {
    ops.Correlation: compile_corr,
    ops.StandardDev: _variance_like('stddev'),
    ops.Variance: _variance_like('var'),
    ops.Covariance: compile_cov,
}

# TRIGONOMETRIC
_trigonometric_ops = {
    ops.Acos: unary('acos'),
    ops.Asin: unary('asin'),
    ops.Atan: unary('atan'),
    ops.Atan2: fixed_arity('atan2', 2),
    ops.Cos: unary('cos'),
    ops.Cot: unary('cot'),
    ops.Sin: unary('sin'),
    ops.Tan: unary('tan')
}

_geometric_ops = {
    Distance_In_Meters: fixed_arity('distance_in_meters', 4),
    Conv_4326_900913_X: unary('conv_4326_900913_x'),
    Conv_4326_900913_Y: unary('conv_4326_900913_y')
}


class ByteLength(ops.StringLength):
    """Returns the length of a string in bytes length"""


_string_ops = {
    ops.StringLength: compile_length(),
    ByteLength: compile_length('byte_length', 'LENGTH'),
    ops.RegexSearch: binary_infix_op('REGEXP'),
    ops.StringSQLLike: binary_infix_op('like'),
    ops.StringSQLILike: binary_infix_op('ilike'),
}

_date_part_truncate = [
    'YEAR', 'QUARTER', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
    'MILLENNIUM', 'CENTURY', 'DECADE', 'WEEK', 'QUARTERDAY'
]

_date_part_extract = [
    'YEAR', 'QUARTER', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
    'DOW', 'ISODOW', 'DOY', 'EPOCH', 'QUARTERDAY', 'WEEK'

]

_date_part_datediff = [
    'YEAR', 'QUARTER', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
    'MILLENNIUM', 'CENTURY', 'DECADE', 'WEEK', 'QUARTERDAY'
]

_interval_dateadd = [
    'YEAR', 'QUARTER', 'MONTH', 'DAYOFYEAR', 'DAY', 'WEEK', 'WEEKDAY', 'HOUR',
    'MINUTE', 'SECOND', 'MILLISECOND'
]
_interval_datepart = [
    'YEAR', 'QUARTER', 'MONTH', 'DAYOFYEAR', 'DAY', 'WEEK', 'WEEKDAY', 'HOUR',
    'MINUTE', 'SECOND', 'MILLISECOND'
]

timestamp_code = {
    'Y': 'YEAR',
    'M': 'MONTH',
    'D': 'DAY',
    'W': 'WEEK',
    'Q': 'QUARTER',
    'h': 'HOUR',
    'm': 'MINUTE',
    's': 'SECOND',
}

_date_ops = {
    ops.Date: unary('toDate'),
    DateTruncate: date_truncate,
    ops.TimestampNow: fixed_arity('NOW', 0),
    TimestampTruncate: date_truncate,

    # DIRECT EXTRACT OPERATIONS
    ops.ExtractYear: _extract_field('YEAR'),
    ops.ExtractMonth: _extract_field('MONTH'),
    ops.ExtractDay: _extract_field('DAY'),
    ops.ExtractHour: _extract_field('HOUR'),
    ops.ExtractMinute: _extract_field('MINUTE'),
    ops.ExtractSecond: _extract_field('SECOND'),

    TimestampExtract: timestamp_binary_infix_op(
        'EXTRACT', 'FROM', timestamp_code=timestamp_code
    ),

    ops.IntervalAdd: _interval_from_integer,
    ops.IntervalFromInteger: _interval_from_integer,

    ops.DateAdd: _timestamp_op('TIMESTAMPADD'),
    ops.DateSub: _timestamp_op('TIMESTAMPADD', '-'),
    ops.TimestampAdd: _timestamp_op('TIMESTAMPADD'),
    ops.TimestampSub: _timestamp_op('TIMESTAMPADD', '-'),
}


class ApproxCountDistinct(ops.Reduction):
    """Approximate number of unique values

    """
    arg = ops.Arg(rlz.column(rlz.any))
    approx = ops.Arg(rlz.integer, default=1)
    where = ops.Arg(rlz.boolean, default=None)

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        # return ir.DoubleScalar
        return ops.partial(ir.IntegerScalar, dtype=ops.dt.int64)


approx_count_distinct = _reduction(
    'approx_nunique',
    sql_func_name='approx_count_distinct',
    sql_signature='{}({})'
)

count_distinct = _reduction('count')
count = _reduction('count')

_agg_ops = {
    ops.Count: count,
    ops.CountDistinct: count_distinct,
    ApproxCountDistinct: approx_count_distinct,
    ops.DistinctColumn: unary_prefix_op('distinct'),
    ops.Mean: _reduction('avg'),
    ops.Max: _reduction('max'),
    ops.Min: _reduction('min'),
    ops.Sum: _reduction('sum'),
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


def assign_functions_to_dtype(dtype, function_ops, forced=False):
    """

    :param dtype:
    :param function_ops: dict
    :param forced:
    :return:
    """
    for klass in function_ops.keys():
        # skip if the class is already in the ibis operations
        if klass in ops.__dict__.values() and not forced:
            continue
        # assign new function to the defined DataType
        func_name = _operation_registry[klass].__name__
        _add_method(dtype, klass, func_name)


def _add_method(dtype, klass, func_name):
    """

    :param dtype:
    :param klass:
    :param func_name:
    :return:
    """
    def f(_klass):
        """
        Return a lambda function that return to_expr() result from the
        custom classes.
        """
        def _f(*args, **kwargs):
            return _klass(*args, **kwargs).to_expr()
        return _f
    # assign new function to the defined DataType
    setattr(
        dtype, func_name, f(klass)
    )


# numeric operations
assign_functions_to_dtype(ir.NumericValue, _trigonometric_ops, forced=True)
assign_functions_to_dtype(ir.NumericValue, _math_ops, forced=True)
assign_functions_to_dtype(ir.NumericValue, _geometric_ops, forced=True)
assign_functions_to_dtype(ir.NumericValue, _stats_ops, forced=False)
assign_functions_to_dtype(ir.ColumnExpr, _agg_ops, forced=True)
# string operations
assign_functions_to_dtype(ir.StringValue, _string_ops, forced=True)

# date/time/timestamp operations
assign_functions_to_dtype(ir.TimestampColumn, _date_ops, forced=True)
assign_functions_to_dtype(ir.DateColumn, _date_ops, forced=True)

_add_method(ir.TimestampColumn, TimestampTruncate, 'truncate')
_add_method(ir.DateColumn, DateTruncate, 'truncate')
_add_method(ir.TimestampColumn, TimestampExtract, 'extract')
# _add_method(ir.DateColumn, TimestampExtract, 'extract')

