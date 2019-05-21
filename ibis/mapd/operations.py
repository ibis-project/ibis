from datetime import date, datetime
from ibis.mapd.identifiers import quote_identifier, _identifiers
from ibis.impala import compiler as impala_compiler
from six import StringIO


import ibis
import ibis.common as com
import ibis.util as util
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
import ibis.expr.operations as ops

_sql_type_names = {
    'int8': 'smallint',
    'int16': 'smallint',
    'int32': 'int',
    'int64': 'bigint',
    'float': 'float',
    'float64': 'double',
    'double': 'double',
    'string': 'text',
    'boolean': 'boolean',
    'timestamp': 'timestamp',
    'decimal': 'decimal',
    'date': 'date',
    'time': 'time',
}


def _is_floating(*args):
    for arg in args:
        if isinstance(arg, ir.FloatingColumn):
            return True
    return False


def _type_to_sql_string(tval):
    if isinstance(tval, dt.Decimal):
        return 'decimal({}, {})'.format(tval.precision, tval.scale)
    else:
        return _sql_type_names[tval.name.lower()]


def _cast(translator, expr):
    from ibis.mapd.client import MapDDataType

    op = expr.op()
    arg, target = op.args
    arg_ = translator.translate(arg)
    type_ = str(MapDDataType.from_ibis(target, nullable=False))

    return 'CAST({0!s} AS {1!s})'.format(arg_, type_)


def _all(expr):
    op = expr.op()
    arg = op.args[0]

    if isinstance(arg, ir.BooleanValue):
        arg = arg.ifelse(1, 0)

    return (1 - arg).sum() == 0


def _any(expr):
    op = expr.op()
    arg = op.args[0]

    if isinstance(arg, ir.BooleanValue):
        arg = arg.ifelse(1, 0)

    return arg.sum() >= 0


def _not_any(expr):
    op = expr.op()
    arg = op.args[0]

    if isinstance(arg, ir.BooleanValue):
        arg = arg.ifelse(1, 0)

    return arg.sum() == 0


def _not_all(expr):
    op = expr.op()
    arg = op.args[0]

    if isinstance(arg, ir.BooleanValue):
        arg = arg.ifelse(1, 0)

    return (1 - arg).sum() != 0


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

        return _reduction_format(
            translator, variants[how].upper(),
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


def _call(translator, func, *args):
    args_ = ', '.join(map(translator.translate, args))
    return '{0!s}({1!s})'.format(func, args_)


def _extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.args[0])
        return 'EXTRACT({} FROM {})'.format(sql_attr, arg)
    return extract_field_formatter


# STATS

def _corr(translator, expr):
    # pull out the arguments to the expression
    args = expr.op().args

    x, y, how, where = args

    # compile the argument
    compiled_x = translator.translate(x)
    compiled_y = translator.translate(y)

    return 'CORR({}, {})'.format(compiled_x, compiled_y)


def _cov(translator, expr):
    # pull out the arguments to the expression
    args = expr.op().args

    x, y, how, where = args

    # compile the argument
    compiled_x = translator.translate(x)
    compiled_y = translator.translate(y)

    return 'COVAR_{}({}, {})'.format(
        how[:4].upper(), compiled_x, compiled_y
    )


# STRING

def _length(func_name='length', sql_func_name='CHAR_LENGTH'):
    def __lenght(translator, expr):
        # pull out the arguments to the expression
        arg = expr.op().args[0]
        # compile the argument
        compiled_arg = translator.translate(arg)
        return '{}({})'.format(sql_func_name, compiled_arg)
    __lenght.__name__ = func_name
    return __lenght


def _contains(translator, expr):
    arg, pattern = expr.op().args[:2]

    pattern_ = '%{}%'.format(translator.translate(pattern)[1:-1])

    return _parenthesize(translator, arg.like(pattern_).ifelse(1, -1))


# GENERIC

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


def raise_unsupported_expr_error(expr):
    msg = "MapD backend doesn't support {} operation!"
    op = expr.op()
    raise com.UnsupportedOperationError(msg.format(type(op)))


def raise_unsupported_op_error(translator, expr, *args):
    msg = "MapD backend doesn't support {} operation!"
    op = expr.op()
    raise com.UnsupportedOperationError(msg.format(type(op)))


# translator
def _name_expr(formatted_expr, quoted_name):
    if quoted_name in _identifiers:
        quoted_name = '"{}"'.format(quoted_name)
    return '{} AS {}'.format(formatted_expr, quoted_name)


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


def _table_array_view(translator, expr):
    ctx = translator.context
    table = expr.op().table
    query = ctx.get_compiled_expr(table)
    return '(\n{0}\n)'.format(util.indent(query, ctx.indent))


def _timestamp_truncate(translator, expr):
    op = expr.op()
    arg, unit = op.args

    unit_ = dt.Interval(unit=unit).resolution.upper()

    # return _call_date_trunc(translator, converter, arg)
    arg_ = translator.translate(arg)
    return 'DATE_TRUNC({0!s}, {1!s})'.format(unit_, arg_)


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


# AGGREGATION

approx_count_distinct = _reduction(
    'approx_nunique',
    sql_func_name='approx_count_distinct',
    sql_signature='{}({}, 100)'
)

count_distinct = _reduction('count')
count = _reduction('count')


# MATH

class NumericTruncate(ops.NumericBinaryOp):
    """Truncates x to y decimal places"""
    output_type = rlz.shape_like('left', ops.dt.float)


# GEOMETRIC

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


# String

class ByteLength(ops.StringLength):
    """Returns the length of a string in bytes length"""


# https://www.mapd.com/docs/latest/mapd-core-guide/dml/
_binary_infix_ops = {
    # math
    ops.Power: fixed_arity('power', 2),
    ops.NotEquals: impala_compiler._binary_infix_op('<>'),
}

_unary_ops = {}

# COMPARISON
_comparison_ops = {}


# MATH
_math_ops = {
    ops.Degrees: unary('degrees'),  # MapD function
    ops.Modulus: fixed_arity('mod', 2),
    ops.Pi: fixed_arity('pi', 0),
    ops.Radians: unary('radians'),
    NumericTruncate: fixed_arity('truncate', 2)
}

# STATS
_stats_ops = {
    ops.Correlation: _corr,
    ops.StandardDev: _variance_like('stddev'),
    ops.Variance: _variance_like('var'),
    ops.Covariance: _cov,
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

# GEOMETRIC
_geometric_ops = {
    Conv_4326_900913_X: unary('conv_4326_900913_x'),
    Conv_4326_900913_Y: unary('conv_4326_900913_y')
}

# STRING
_string_ops = {
    ops.StringLength: _length(),
    ByteLength: _length('byte_length', 'LENGTH'),
    ops.StringSQLILike: binary_infix_op('ilike'),
    ops.StringFind: _contains
}

# DATE
_date_ops = {
    ops.DateTruncate: _timestamp_truncate,
    ops.TimestampTruncate: _timestamp_truncate,

    # DIRECT EXTRACT OPERATIONS
    ops.ExtractYear: _extract_field('YEAR'),
    ops.ExtractMonth: _extract_field('MONTH'),
    ops.ExtractDay: _extract_field('DAY'),
    ops.ExtractHour: _extract_field('HOUR'),
    ops.ExtractMinute: _extract_field('MINUTE'),
    ops.ExtractSecond: _extract_field('SECOND'),

    ops.IntervalAdd: _interval_from_integer,
    ops.IntervalFromInteger: _interval_from_integer,

    ops.DateAdd: _timestamp_op('TIMESTAMPADD'),
    ops.DateSub: _timestamp_op('TIMESTAMPADD', '-'),
    ops.TimestampAdd: _timestamp_op('TIMESTAMPADD'),
    ops.TimestampSub: _timestamp_op('TIMESTAMPADD', '-'),
}

# AGGREGATION/REDUCTION
_agg_ops = {
    ops.HLLCardinality: approx_count_distinct,
    ops.DistinctColumn: unary_prefix_op('distinct'),
}

# GENERAL
_general_ops = {
    ops.Literal: literal,
    ops.ValueList: _value_list,
    ops.Cast: _cast,
    ops.Where: fixed_arity('if', 3),
    ops.TableColumn: _table_column,
}

# UNSUPPORTED OPERATIONS
_unsupported_ops = [
    # generic/aggregation
    ops.CMSMedian,
    ops.WindowOp,
    ops.DecimalPrecision,
    ops.DecimalScale,
    ops.BaseConvert,
    ops.CumulativeSum,
    ops.CumulativeMin,
    ops.CumulativeMax,
    ops.CumulativeMean,
    ops.CumulativeAny,
    ops.CumulativeAll,
    ops.IdenticalTo,
    ops.Arbitrary,
    ops.RowNumber,
    ops.DenseRank,
    ops.MinRank,
    ops.PercentRank,
    ops.FirstValue,
    ops.LastValue,
    ops.NthValue,
    ops.Lag,
    ops.Lead,
    ops.NTile,
    ops.GroupConcat,
    ops.Arbitrary,
    ops.NullIf,
    ops.NullIfZero,
    ops.NullLiteral,
    ops.IsInf,
    ops.IsNan,
    ops.IfNull,
    # string
    ops.Lowercase,
    ops.Uppercase,
    ops.FindInSet,
    ops.StringReplace,
    ops.StringJoin,
    ops.StringSplit,
    ops.Translate,
    ops.StringAscii,
    ops.LPad,
    ops.RPad,
    ops.Strip,
    ops.RStrip,
    ops.LStrip,
    ops.Capitalize,
    ops.Substring,
    ops.StrRight,
    ops.Repeat,
    ops.Reverse,
    ops.RegexExtract,
    ops.RegexReplace,
    ops.ParseURL,
    # Numeric
    ops.Least,
    ops.Greatest,
    ops.Log2,
    ops.Log,
    ops.Round,
    # date/time/timestamp
    ops.TimestampFromUNIX,
    ops.Date,
    ops.TimeTruncate,
    ops.TimestampDiff,
    ops.DayOfWeekIndex,
    ops.DayOfWeekName,
]

_unsupported_ops = {k: raise_unsupported_op_error for k in _unsupported_ops}

# registry
_operation_registry = impala_compiler._operation_registry.copy()

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
_operation_registry.update(_unsupported_ops)
