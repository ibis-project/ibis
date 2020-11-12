"""OmniSciDB operations module."""
import warnings
from datetime import date, datetime
from io import StringIO
from typing import Callable

import ibis
import ibis.common.exceptions as com
import ibis.common.geospatial as geo
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
import ibis.util as util
from ibis import literal as L
from ibis.backends.base_sql import (
    cumulative_to_window,
    format_window,
    operation_registry,
    time_range_to_range_window,
)

from . import dtypes as omniscidb_dtypes
from .identifiers import quote_identifier

_sql_type_names = omniscidb_dtypes.ibis_dtypes_str_to_sql


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
    from .client import OmniSciDBDataType

    op = expr.op()
    arg, target = op.args
    arg_ = translator.translate(arg)

    if isinstance(arg, ir.GeoSpatialValue):
        # NOTE: CastToGeography expects geometry with SRID=4326
        type_ = target.geotype.upper()

        if type_ == 'GEOMETRY':
            raise com.UnsupportedOperationError(
                'OmnisciDB/OmniSciDB doesn\'t support yet convert '
                + 'from GEOGRAPHY to GEOMETRY.'
            )
    else:
        type_ = str(OmniSciDBDataType.from_ibis(target, nullable=False))
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


def fixed_arity(func_name: str, arity: int) -> Callable:
    """Create a translator for a given arity.

    Parameters
    ----------
    func_name : str
    arity : 1

    Returns
    -------
    function

    Raises
    ------
    com.UnsupportedOperationError
        If the arity is not compatible if the parameters of the expression.
    """
    # formatter function
    def formatter(translator, expr):
        op = expr.op()
        arg_count = len(op.args)
        if arity != arg_count:
            msg = 'Incorrect number of args {0} instead of {1}'
            raise com.UnsupportedOperationError(msg.format(arg_count, arity))
        return _call(translator, func_name, *op.args)

    formatter.__name__ = func_name
    return formatter


def unary(func_name: str) -> Callable:
    """Create a translator for a unary operation.

    Parameters
    ----------
    func_name : str

    Returns
    -------
    function
    """
    return fixed_arity(func_name, 1)


def _reduction_format(
    translator,
    func_name,
    sql_func_name=None,
    sql_signature='{}({})',
    arg=None,
    args=None,
    where=None,
):
    if not sql_func_name:
        sql_func_name = func_name

    if where is not None:
        arg = where.ifelse(arg, ibis.NA)

    return sql_signature.format(
        sql_func_name, ', '.join(map(translator.translate, [arg] + list(args)))
    )


def _reduction(func_name, sql_func_name=None, sql_signature='{}({})'):
    def formatter(translator, expr):
        op = expr.op()

        # HACK: support trailing arguments
        where = op.where
        args = []

        for arg in op.args:
            if arg is not where:
                if arg.type().equals(dt.boolean):
                    arg = arg.ifelse(1, 0)
                args.append(arg)

        return _reduction_format(
            translator,
            func_name,
            sql_func_name,
            sql_signature,
            args[0],
            args[1:],
            where,
        )

    formatter.__name__ = func_name
    return formatter


def _variance_like(func):
    variants = {'sample': '{}_SAMP'.format(func), 'pop': '{}_POP'.format(func)}

    def formatter(translator, expr):
        arg, how, where = expr.op().args

        return _reduction_format(
            translator, variants[how].upper(), None, '{}({})', arg, [], where
        )

    formatter.__name__ = func
    return formatter


def unary_prefix_op(prefix_op: str) -> Callable:
    """Create a unary translator with a prefix.

    Parameters
    ----------
    prefix_op : str

    Returns
    -------
    function
    """
    # formatter function
    def formatter(translator, expr):
        op = expr.op()
        arg = _parenthesize(translator, op.args[0])

        return '{0!s} {1!s}'.format(prefix_op.upper(), arg)

    formatter.__name__ = prefix_op
    return formatter


def binary_infix_op(infix_sym: str) -> Callable:
    """Create a binary infix translator.

    Parameters
    ----------
    infix_sym : str

    Returns
    -------
    function
    """
    # formatter function
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
    def extract_field_formatter(translator, expr, sql_attr=sql_attr):
        adjustments = {
            'MILLISECOND': 1000,
        }
        adjustment = adjustments.get(sql_attr, None)

        op = expr.op()
        arg = op.args[0]

        arg_str = translator.translate(arg)
        result = 'EXTRACT({} FROM {})'.format(sql_attr, arg_str)

        if sql_attr == 'ISODOW':
            result += '- 1'

        if adjustment:
            # used by time extraction
            result = ' mod({}, {})'.format(result, adjustment)

        return result

    return extract_field_formatter


def _extract_field_dow_name(sql_attr):
    def extract_field_formatter(translator, expr, sql_attr=sql_attr):
        op = expr.op()
        week_names = [
            'Monday',
            'Tuesday',
            'Wednesday',
            'Thursday',
            'Friday',
            'Saturday',
            'Sunday',
        ]

        expr_new = ops.DayOfWeekIndex(op.args[0]).to_expr()
        expr_new = expr_new.case()
        for i in range(7):
            expr_new = expr_new.when(i, week_names[i])
        expr_new = expr_new.else_('').end()

        return translator.translate(expr_new)

    return extract_field_formatter


# MATH


def _log_common(translator, arg, base=None):
    if isinstance(arg, tuple):
        args_ = ', '.join(map(translator.translate, arg))
    else:
        args_ = translator.translate(arg)

    if base is None:
        return 'ln({})'.format(args_)

    base_formatted = translator.translate(base)
    return 'ln({})/ln({})'.format(args_, base_formatted)


def _log(translator, expr):
    op = expr.op()
    arg, base = op.args
    return _log_common(translator, arg, base)


def _log2(translator, expr):
    op = expr.op()
    arg = op.args
    return _log_common(translator, arg, L(2))


def _log10(translator, expr):
    op = expr.op()
    arg = op.args
    return _log_common(translator, arg, L(10))


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

    return 'COVAR_{}({}, {})'.format(how[:4].upper(), compiled_x, compiled_y)


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
            "OmniSciDB doesn't support subsecond interval resolutions"
        )

    return '{1}, (sign){0}'.format(expr.op().value, dtype.resolution.upper())


def _interval_from_integer(translator, expr):
    op = expr.op()
    arg, unit = op.args

    dtype = expr.type()
    if dtype.unit in {'ms', 'us', 'ns'}:
        raise com.UnsupportedOperationError(
            "OmniSciDB doesn't support subsecond interval resolutions"
        )

    arg_ = translator.translate(arg)
    return '{}, (sign){}'.format(dtype.resolution.upper(), arg_)


def _todate(translator, expr):
    op = expr.op()
    arg_expr = op.args
    if isinstance(arg_expr, tuple):
        arg = ', '.join(map(translator.translate, arg_expr))
    else:
        arg = translator.translate(arg_expr)
    return 'CAST({} AS date)'.format(arg)


def _timestamp_op(func, op_sign='+'):
    def _formatter(translator, expr):
        op = expr.op()
        left, right = op.args

        formatted_left = translator.translate(left)
        formatted_right = translator.translate(right)

        if isinstance(left, ir.DateValue):
            formatted_left = 'CAST({} as timestamp)'.format(formatted_left)

        return '{}({}, {})'.format(
            func, formatted_right.replace('(sign)', op_sign), formatted_left
        )

    return _formatter


def _timestampdiff(date_part='second', op_sign='+'):
    def _formatter(translator, expr):
        op = expr.op()
        left, right = op.args

        formatted_left = translator.translate(left)
        formatted_right = translator.translate(right)

        if isinstance(left, (ir.DateValue, ir.TimestampValue)):
            formatted_left = 'CAST({} as timestamp)'.format(formatted_left)
        if isinstance(right, (ir.DateValue, ir.TimestampValue)):
            formatted_right = 'CAST({} as timestamp)'.format(formatted_right)

        return "timestampdiff({}, {}, {})".format(
            date_part,
            formatted_right.replace('(sign)', op_sign),
            formatted_left,
        )

    return _formatter


def _datadiff(date_part='day', op_sign='+'):
    def _formatter(translator, expr):
        op = expr.op()
        left, right = op.args

        formatted_left = translator.translate(left)
        formatted_right = translator.translate(right)

        if isinstance(left, (ir.DateValue, ir.TimestampValue)):
            formatted_left = 'CAST({} as date)'.format(formatted_left)
        if isinstance(right, (ir.DateValue, ir.TimestampValue)):
            formatted_right = 'CAST({} as date)'.format(formatted_right)

        return "datediff('{}', {}, {})".format(
            date_part,
            formatted_right.replace('(sign)', op_sign),
            formatted_left,
        )

    return _formatter


def _set_literal_format(translator, expr):
    value_type = expr.type().value_type

    formatted = [
        translator.translate(ir.literal(x, type=value_type))
        for x in expr.op().value
    ]

    return '({})'.format(', '.join(formatted))


def _cross_join(translator, expr):
    args = expr.op().args
    left, right = args[:2]
    return translator.translate(left.join(right, ibis.literal(True)))


def _ifnull(translator, expr):
    col_expr, value_expr = expr.op().args
    if isinstance(col_expr, ir.DecimalValue) and isinstance(
        value_expr, ir.IntegerValue
    ):
        value_expr = value_expr.cast(col_expr.type())
    col_name = translator.translate(col_expr)
    value = translator.translate(value_expr)
    return 'CASE WHEN {} IS NULL THEN {} ELSE {} END'.format(
        col_name, value, col_name
    )


def _nullifzero(translator, expr):
    col_expr = expr.op().args[0]
    return translator.translate(col_expr.nullif(0))


def _zeroifnull(translator, expr):
    col_expr = expr.op().args[0]
    return translator.translate(col_expr.fillna(0))


def literal(translator, expr: ibis.expr.operations.Literal) -> str:
    """Create a translator for literal operations.

    Parameters
    ----------
    translator : ibis.omniscidb.compiler.OmniSciDBExprTranslator
    expr : ibis.expr.operations.Literal

    Returns
    -------
    translated : str

    Raises
    ------
    Exception
        if a TimestampValue expr is given and its value is a datetime and
        the format is invalid.
    NotImplementedError
        if the given literal expression is not recognized.
    """
    op = expr.op()
    value = op.value

    # geo spatial data type
    if isinstance(expr, ir.GeoSpatialScalar):
        return geo.translate_literal(expr)
    # primitive data type
    elif isinstance(expr, ir.BooleanValue):
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
                warnings.warn(msg.format(value))
            value = value.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(value, str):
            # check if the datetime format is a valid format (
            # '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d'). if format is '%Y-%m-%d' it
            # is converted to '%Y-%m-%d 00:00:00'
            msg = (
                "Literal datetime string should use '%Y-%m-%d %H:%M:%S' "
                "format. When '%Y-%m-%d' format is used,  datetime will be "
                "converted automatically to '%Y-%m-%d 00:00:00'"
            )

            try:
                dt_value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    dt_value = datetime.strptime(value, '%Y-%m-%d')
                    warnings.warn(msg)
                except ValueError:
                    raise Exception(msg)

            value = dt_value.strftime('%Y-%m-%d %H:%M:%S')

        return "'{0!s}'".format(value)
    elif isinstance(expr, ir.DateValue):
        if isinstance(value, date):
            value = value.strftime('%Y-%m-%d')
        return "CAST('{0!s}' as date)".format(value)
    # array data type
    elif isinstance(expr, ir.ArrayValue):
        return str(list(value))
    else:
        raise NotImplementedError(type(expr))


def _where(translator, expr):
    # pull out the arguments to the expression
    args = expr.op().args
    condition, expr1, expr2 = args
    expr = condition.ifelse(expr1, expr2)
    return translator.translate(expr)


def raise_unsupported_expr_error(expr: ibis.Expr):
    """Raise an unsupported expression error for given expression.

    Parameters
    ----------
    expr : ibis.Expr

    Raises
    ------
    com.UnsupportedOperationError
    """
    msg = "OmniSciDB backend doesn't support {} operation!"
    op = expr.op()
    raise com.UnsupportedOperationError(msg.format(type(op)))


def raise_unsupported_op_error(translator, expr, *args):
    """Raise an unsupported operation error for given expression.

    Parameters
    ----------
    expr : ibis.Expr

    Raises
    ------
    com.UnsupportedOperationError
    """
    msg = "OmniSciDB backend doesn't support {} operation!"
    op = expr.op()
    raise com.UnsupportedOperationError(msg.format(type(op)))


# translator
def _name_expr(formatted_expr, quoted_name):
    return '{} AS {}'.format(formatted_expr, quote_identifier(quoted_name))


class CaseFormatter:
    """Case Formatter class."""

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
        """Return the CASE statement formatted.

        Results
        -------
        str
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
    sql_signature='{}({}, 100)',
)

count_distinct = _reduction('count')
count = _reduction('count')


def _arbitrary(translator, expr):
    arg, how, where = expr.op().args

    if how not in (None, 'last'):
        raise com.UnsupportedOperationError(
            '{!r} value not supported for arbitrary in OmniSciDB'.format(how)
        )

    if where is not None:
        arg = where.ifelse(arg, ibis.NA)

    return 'SAMPLE({})'.format(translator.translate(arg))


# MATH


class NumericTruncate(ops.NumericBinaryOp):
    """Truncates x to y decimal places."""

    output_type = rlz.shape_like('left', ops.dt.float)


# GEOMETRIC


class Conv_4326_900913_X(ops.UnaryOp):
    """Converts WGS-84 latitude to WGS-84 Web Mercator x coordinate."""

    output_type = rlz.shape_like('arg', ops.dt.float)


class Conv_4326_900913_Y(ops.UnaryOp):
    """Converts WGS-84 longitude to WGS-84 Web Mercator y coordinate."""

    output_type = rlz.shape_like('arg', ops.dt.float)


# String


class ByteLength(ops.StringLength):
    """Returns the length of a string in bytes length."""


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

    _unsupported_win_ops = (
        ops.CMSMedian,
        ops.GroupConcat,
        ops.HLLCardinality,
        ops.All,  # TODO: change all to work as cumall
        ops.Any,  # TODO: change any to work as cumany
    )

    _subtract_one = '{} - 1'.format
    _expr_transforms = {
        ops.DenseRank: _subtract_one,
        ops.MinRank: _subtract_one,
        ops.NTile: _subtract_one,
        ops.RowNumber: _subtract_one,
    }

    if isinstance(window_op, _unsupported_win_ops):
        raise com.UnsupportedOperationError(
            '{} is not supported in window functions'.format(type(window_op))
        )

    if isinstance(window_op, ops.CumulativeOp):
        arg = cumulative_to_window(translator, arg, window)
        return translator.translate(arg)

    if window.preceding is not None:
        raise com.UnsupportedOperationError(
            'Window preceding is not supported by OmniSciDB backend yet'
        )

    if window.following is not None and window.following != 0:
        raise com.UnsupportedOperationError(
            'Window following is not supported by OmniSciDB backend yet'
        )
    window.following = None

    # Some analytic functions need to have the expression of interest in
    # the ORDER BY part of the window clause
    if isinstance(window_op, _require_order_by) and len(window._order_by) == 0:
        window = window.order_by(window_op.args[0])

    # Time ranges need to be converted to microseconds.
    if window.how == 'range':
        order_by_types = [type(x.op().args[0]) for x in window._order_by]
        time_range_types = (ir.TimeColumn, ir.DateColumn, ir.TimestampColumn)
        if any(col_type in time_range_types for col_type in order_by_types):
            window = time_range_to_range_window(translator, window)

    window_formatted = format_window(translator, op, window)

    arg_formatted = translator.translate(arg)
    result = '{} {}'.format(arg_formatted, window_formatted)

    if type(window_op) in _expr_transforms:
        return _expr_transforms[type(window_op)](result)
    else:
        return result


def _shift_like(name, default_offset=None):
    def formatter(translator, expr):
        op = expr.op()
        arg, offset, default = op.args

        arg_formatted = translator.translate(arg)

        if default is not None:
            if offset is None:
                offset_formatted = (
                    '1' if default_offset is None else str(default_offset)
                )
            else:
                offset_formatted = translator.translate(offset)

            default_formatted = translator.translate(default)

            return '{}({}, {}, {})'.format(
                name, arg_formatted, offset_formatted, default_formatted
            )
        elif offset is not None or default_offset is not None:
            offset_formatted = (
                translator.translate(offset)
                if offset is not None
                else str(default_offset)
            )
            return '{}({}, {})'.format(name, arg_formatted, offset_formatted)
        else:
            return '{}({})'.format(name, arg_formatted)

    return formatter


def _window_op_one_param(name):
    def formatter(translator, expr):
        op = expr.op()
        _, parameter = op.args
        # arg_formatted = translator.translate(arg)
        parameter_formatted = translator.translate(parameter)
        return '{}({})'.format(name, parameter_formatted)

    return formatter


# operation map

_binary_infix_ops = {
    # math
    ops.Power: fixed_arity('power', 2),
    ops.NotEquals: binary_infix_op('<>'),
}

_unary_ops = {}

# COMPARISON
_comparison_ops = {}


# MATH
_math_ops = {
    ops.Degrees: unary('degrees'),  # OmniSciDB function
    ops.Modulus: fixed_arity('mod', 2),
    ops.Pi: fixed_arity('pi', 0),
    ops.Radians: unary('radians'),
    NumericTruncate: fixed_arity('truncate', 2),
    ops.Log: _log,
    ops.Log2: _log2,
    ops.Log10: _log10,
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
    ops.Tan: unary('tan'),
}

# GEOMETRIC
_geometric_ops = {
    Conv_4326_900913_X: unary('conv_4326_900913_x'),
    Conv_4326_900913_Y: unary('conv_4326_900913_y'),
}

# GEO SPATIAL
_geospatial_ops = {
    ops.GeoArea: unary('ST_AREA'),
    ops.GeoContains: fixed_arity('ST_CONTAINS', 2),
    ops.GeoDistance: fixed_arity('ST_DISTANCE', 2),
    ops.GeoDisjoint: fixed_arity('ST_DISJOINT', 2),
    ops.GeoDFullyWithin: fixed_arity('ST_DFULLYWITHIN', 3),
    ops.GeoDWithin: fixed_arity('ST_DWITHIN', 3),
    ops.GeoEndPoint: unary('ST_ENDPOINT'),
    ops.GeoIntersects: fixed_arity('ST_INTERSECTS', 2),
    ops.GeoLength: unary('ST_LENGTH'),
    ops.GeoMaxDistance: fixed_arity('ST_MAXDISTANCE', 2),
    ops.GeoNPoints: unary('ST_NPOINTS'),
    ops.GeoNRings: unary('ST_NRINGS'),
    ops.GeoPerimeter: unary('ST_PERIMETER'),
    ops.GeoPoint: fixed_arity('ST_POINT', 2),
    ops.GeoPointN: fixed_arity('ST_POINTN', 2),
    ops.GeoSetSRID: fixed_arity('ST_SETSRID', 2),
    ops.GeoSRID: unary('ST_SRID'),
    ops.GeoStartPoint: unary('ST_STARTPOINT'),
    ops.GeoTransform: fixed_arity('ST_TRANSFORM', 2),
    ops.GeoWithin: fixed_arity('ST_WITHIN', 2),
    ops.GeoX: unary('ST_X'),
    ops.GeoY: unary('ST_Y'),
    ops.GeoXMin: unary('ST_XMIN'),
    ops.GeoXMax: unary('ST_XMAX'),
    ops.GeoYMin: unary('ST_YMIN'),
    ops.GeoYMax: unary('ST_YMAX'),
}

# STRING
_string_ops = {
    ops.StringLength: _length(),
    ByteLength: _length('byte_length', 'LENGTH'),
    ops.StringSQLILike: binary_infix_op('ilike'),
    ops.StringFind: _contains,
}

# DATE
_date_ops = {
    ops.DateTruncate: _timestamp_truncate,
    ops.TimestampTruncate: _timestamp_truncate,
    # DIRECT EXTRACT OPERATIONS
    ops.ExtractYear: _extract_field('YEAR'),
    ops.ExtractMonth: _extract_field('MONTH'),
    ops.ExtractDay: _extract_field('DAY'),
    ops.ExtractDayOfYear: _extract_field('DOY'),
    ops.ExtractQuarter: _extract_field('QUARTER'),
    ops.DayOfWeekIndex: _extract_field('ISODOW'),
    ops.DayOfWeekName: _extract_field_dow_name('ISODOW'),
    ops.ExtractEpochSeconds: _extract_field('EPOCH'),
    ops.ExtractHour: _extract_field('HOUR'),
    ops.ExtractMinute: _extract_field('MINUTE'),
    ops.ExtractSecond: _extract_field('SECOND'),
    ops.ExtractMillisecond: _extract_field('MILLISECOND'),
    ops.IntervalAdd: _interval_from_integer,
    ops.IntervalFromInteger: _interval_from_integer,
    ops.DateAdd: _timestamp_op('TIMESTAMPADD'),
    ops.DateSub: _timestamp_op('TIMESTAMPADD', '-'),
    ops.TimestampAdd: _timestamp_op('TIMESTAMPADD'),
    ops.TimestampSub: _timestamp_op('TIMESTAMPADD', '-'),
    ops.TimestampDiff: _timestampdiff(),
    ops.DateDiff: _datadiff(),
    ops.Date: _todate,
}

# AGGREGATION/REDUCTION
_agg_ops = {
    ops.HLLCardinality: approx_count_distinct,
    ops.DistinctColumn: unary_prefix_op('distinct'),
    ops.Arbitrary: _arbitrary,
    ops.Sum: _reduction('sum'),
    ops.Mean: _reduction('avg'),
    ops.Min: _reduction('min'),
    ops.Max: _reduction('max'),
}

# GENERAL
_general_ops = {
    ops.Literal: literal,
    ops.NullLiteral: lambda *args: 'NULL',
    ops.ValueList: _value_list,
    ops.Cast: _cast,
    ops.Where: _where,
    ops.TableColumn: _table_column,
    ops.CrossJoin: _cross_join,
    ops.IfNull: _ifnull,
    ops.NullIf: fixed_arity('nullif', 2),
    ops.IsNan: unary('isNan'),
    ops.NullIfZero: _nullifzero,
    ops.ZeroIfNull: _zeroifnull,
    ops.RowID: lambda *args: 'rowid',
}

# WINDOW
# RowNumber, and rank functions starts with 0 in Ibis-land
_window_ops = {
    ops.DenseRank: lambda *args: 'dense_rank()',
    ops.FirstValue: unary('first_value'),
    ops.LastValue: unary('last_value'),
    ops.Lag: _shift_like('lag'),
    ops.Lead: _shift_like('lead', 1),
    ops.MinRank: lambda *args: 'rank()',
    # cume_dist vs percent_rank
    # https://github.com/ibis-project/ibis/issues/1975
    ops.PercentRank: lambda *args: 'cume_dist()',
    ops.RowNumber: lambda *args: 'row_number()',
    ops.WindowOp: _window,
}

# UNSUPPORTED OPERATIONS
_unsupported_ops = [
    # generic/aggregation
    ops.CMSMedian,
    ops.DecimalPrecision,
    ops.DecimalScale,
    ops.BaseConvert,
    ops.CumulativeAny,
    ops.CumulativeAll,
    ops.IdenticalTo,
    ops.NTile,
    ops.NthValue,
    ops.GroupConcat,
    ops.IsInf,
    # string
    ops.Lowercase,
    ops.Uppercase,
    ops.FindInSet,
    ops.StringReplace,
    ops.StringJoin,
    ops.StringSplit,
    ops.StringToTimestamp,
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
    ops.ExtractWeekOfYear,
    ops.TimestampFromUNIX,
    ops.TimeTruncate,
    # table
    ops.Union,
]

_unsupported_ops = {k: raise_unsupported_op_error for k in _unsupported_ops}

# registry
_operation_registry = {**operation_registry}

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
_operation_registry.update(_geospatial_ops)
_operation_registry.update(_window_ops)
# the last update should be with unsupported ops
_operation_registry.update(_unsupported_ops)
