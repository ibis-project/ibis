'''
Adding and subtracting timestamp/date intervals (dealt with in `_timestamp_op`)
is still WIP since Spark SQL support for these tasks is not comprehensive.
The built-in Spark SQL functions `date_add`, `date_sub`, and `add_months` do
not support timestamps, as they set the HH:MM:SS part to zero. The other option
is arithmetic syntax: <timestamp> + INTERVAL <num> <unit>, where unit is
something like MONTHS or DAYS. However, with the arithmetic syntax, <num>
must be a scalar, i.e. can't be a column like t.int_col.

                        supports        supports        preserves
                        scalars         columns         HH:MM:SS
                     _______________________________ _______________
built-in functions  |               |               |               |
like `date_add`     |      YES      |      YES      |       NO      |
                    |_______________|_______________|_______________|
                    |               |               |               |
arithmetic          |      YES      |       NO      |      YES      |
                    |_______________|_______________|_______________|


OTHER TODO:
  - Fix strftime to be consistent with Ibis standard strftime, instead of
    Java SimpleDateFormat
  - Fix DayOfWeekName rewrite to be consistent with strftime ^^
  - Decide if automatically setting UTC timezone in api.py is desired

'''


import itertools
import math

import ibis
import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.sql.compiler as comp
import ibis.util as util
from ibis.impala import compiler as impala_compiler
from ibis.impala.compiler import (
    ImpalaContext,
    ImpalaDialect,
    ImpalaExprTranslator,
    ImpalaSelect,
    _reduction,
    fixed_arity,
    unary,
)


class SparkSelectBuilder(comp.SelectBuilder):
    @property
    def _select_class(self):
        return SparkSelect


class SparkQueryBuilder(comp.QueryBuilder):
    select_builder = SparkSelectBuilder


def build_ast(expr, context):
    builder = SparkQueryBuilder(expr, context=context)
    return builder.get_result()


class SparkContext(ImpalaContext):
    pass


def _is_inf(translator, expr):
    arg, = expr.op().args
    return '{} = {} or {} = {}'.format(
        *map(
            translator.translate,
            [
                arg,
                ibis.literal(math.inf, type='double'),
                arg,
                ibis.literal(-math.inf, type='double'),
            ]
        )
    )


_sql_type_names = impala_compiler._sql_type_names.copy()

_sql_type_names.update({
    'date': 'date',
})


def _cast(translator, expr):
    op = expr.op()
    arg, target_type = op.args
    arg_formatted = translator.translate(arg)

    if isinstance(arg, ir.CategoryValue) and target_type == dt.int32:
        return arg_formatted
    if isinstance(arg, ir.TemporalValue) and target_type == dt.int64:
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


def _timestamp_op(func):
    def _formatter(translator, expr):
        op = expr.op()
        left, right = op.args
        formatted_left = translator.translate(left)
        formatted_right = translator.translate(right)
        func_name = func

        if isinstance(left, (ir.TimestampScalar, ir.DateValue)):
            formatted_left = 'cast({} as timestamp)'.format(formatted_left)

        if isinstance(right, (ir.TimestampScalar, ir.DateValue)):
            formatted_right = 'cast({} as timestamp)'.format(formatted_right)

        # Spark only supports date_add, date_sub, and add_months.
        # date_add and date_sub are in days, add_months is in months.
        # These ops set HH:MM:SS to zero, operating only on date.
        #
        # If we use the <timestamp> + INTERVAL <int> <unit> syntax,
        # Spark only supports the second argument being a literal.
        if func_name in ('date_add', 'date_sub'):
            # if not isinstance(right, ir.IntervalScalar):
            #     raise NotImplementedError('Interval columns not supported')

            # mostly correct but only works with scalars
            operator = '+' if func_name == 'date_add' else '-'
            return '{} {} {}'.format(formatted_left, operator, formatted_right)

            # incorrect but works with columns too
            if isinstance(right.op(), ops.IntervalFromInteger):
                unit = right.op().unit
                value = right.op().arg.op().value
            elif isinstance(right.op(), ops.Literal):
                unit = right.op().dtype.unit
                value = right.op().value
            if unit in ('Y', 'M'):
                func_name = 'add_months'
                if func_name == 'date_sub':
                    value *= -1
                if unit == 'Y':
                    value *= 12
            elif unit in ('W', 'D'):
                if unit == 'W':
                    value *= 7
            formatted_right = translator.translate(ibis.literal(value))

        return '{}({}, {})'.format(func_name, formatted_left, formatted_right)

    return _formatter


def _timestamp_diff(translator, expr):
    op = expr.op()
    left, right = op.args
    casted_left = 'cast(to_timestamp({}) as double)'.format(
        translator.translate(left)
    )
    casted_right = 'cast(to_timestamp({}) as double)'.format(
        translator.translate(right)
    )

    return '{} - {}'.format(
        casted_left, casted_right
    )


_spark_date_unit_names = {
    'Y': 'YEAR',
    'Q': 'QUARTER',
}

_spark_timestamp_unit_names = _spark_date_unit_names.copy()
_spark_timestamp_unit_names.update({
    'M': 'MONTH',
    'W': 'WEEK',
    'D': 'DAY',
    'h': 'HOUR',
    'm': 'MINUTE',
    's': 'SECOND'
})


def _timestamp_truncate(translator, expr):
    op = expr.op()
    arg, unit = op.args

    arg_formatted = translator.translate(arg)
    try:
        unit = _spark_timestamp_unit_names[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            '{!r} unit is not supported in timestamp truncate'.format(unit)
        )

    if unit == 'DAY':
        return "date(date_trunc('{}', {}))".format(unit, arg_formatted)
    else:
        return "date_trunc('{}', {})".format(unit, arg_formatted)


def _date_truncate(translator, expr):
    op = expr.op()
    arg, unit = op.args

    arg_formatted = translator.translate(arg)
    try:
        unit = _spark_date_unit_names[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            '{!r} unit is not supported in date truncate'.format(unit)
        )

    return "trunc({}, '{}')".format(arg_formatted, unit)


def _timestamp_from_unix(translator, expr):
    op = expr.op()

    val, unit = op.args
    val = util.convert_unit(val, unit, 's', floor=False)

    arg = translator.translate(val)
    return 'to_timestamp({})'.format(arg)


def _string_concat(translator, expr):
    return 'CONCAT({})'.format(
        ', '.join(map(translator.translate, expr.op().arg))
    )


def _array_literal_format(translator, expr):
    translated_values = [
        translator.translate(ibis.literal(x))
        for x in expr.op().value
    ]

    return 'array({})'.format(
        ', '.join(translated_values)
    )


def _struct_like_format(func):
    def formatter(translator, expr):
        translated_values = [
            ("'{}'".format(name), translator.translate(ibis.literal(val)))
            for (name, val) in expr.op().value.items()
        ]

        return '{}({})'.format(
            func,
            ', '.join(itertools.chain(*translated_values))
        )
    return formatter


_literal_formatters = impala_compiler._literal_formatters.copy()

_literal_formatters.update({
    'array': _array_literal_format,
    'struct': _struct_like_format('named_struct'),
    'map': _struct_like_format('map'),
})


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
    elif isinstance(expr, ir.ArrayValue):
        typeclass = 'array'
    elif isinstance(expr, ir.StructScalar):
        typeclass = 'struct'
    elif isinstance(expr, ir.MapScalar):
        typeclass = 'map'
    else:
        raise NotImplementedError(type(expr).__name__)

    return _literal_formatters[typeclass](translator, expr)


def _struct_field(translator, expr):
    arg, field = expr.op().args
    arg_formatted = translator.translate(arg)
    return '{}.`{}`'.format(arg_formatted, field)


def _map_value_for_key(translator, expr):
    arg, field = expr.op().args
    arg_formatted = translator.translate(arg)
    field_formatted = translator.translate(ibis.literal(field))
    return '{}[{}]'.format(arg_formatted, field_formatted)


def _round(translator, expr):
    op = expr.op()
    arg, digits = op.args

    arg_formatted = translator.translate(arg)

    if digits is not None:
        digits_formatted = translator.translate(digits)
        return 'bround({}, {})'.format(arg_formatted, digits_formatted)
    return 'bround({})'.format(arg_formatted)


_operation_registry = impala_compiler._operation_registry.copy()
_operation_registry.update(
    {
        ops.IsNan: unary('isnan'),
        ops.IsInf: _is_inf,
        ops.IfNull: fixed_arity('ifnull', 2),
        ops.StructField: _struct_field,
        ops.MapValueForKey: _map_value_for_key,
        ops.ArrayLength: unary('size'),
        ops.Round: _round,
        ops.HLLCardinality: _reduction('approx_count_distinct'),
        ops.StrRight: fixed_arity('right', 2),
        ops.StringSplit: fixed_arity('SPLIT', 2),
        ops.RegexSearch: fixed_arity('rlike', 2),
        ops.StringConcat: _string_concat,
        ops.ArrayConcat: fixed_arity('concat', 2),
        ops.Cast: _cast,
        ops.ExtractYear: unary('year'),
        ops.ExtractMonth: unary('month'),
        ops.ExtractDay: unary('day'),
        ops.ExtractHour: unary('hour'),
        ops.ExtractMinute: unary('minute'),
        ops.ExtractSecond: unary('second'),
        ops.DateAdd: _timestamp_op('date_add'),
        ops.DateSub: _timestamp_op('date_sub'),
        ops.TimestampAdd: _timestamp_op('date_add'),
        ops.TimestampSub: _timestamp_op('date_sub'),
        ops.TimestampDiff: _timestamp_diff,
        ops.TimestampTruncate: _timestamp_truncate,
        ops.TimestampFromUNIX: _timestamp_from_unix,
        ops.DateTruncate: _date_truncate,
        ops.Literal: _literal,
    }
)


class SparkExprTranslator(ImpalaExprTranslator):
    _registry = _operation_registry

    context_class = SparkContext


compiles = SparkExprTranslator.compiles
rewrites = SparkExprTranslator.rewrites


@compiles(ops.Arbitrary)
def spark_compiles_arbitrary(translator, expr):
    arg, how, where = expr.op().args

    if where is not None:
        arg = where.ifelse(arg, ibis.NA)

    if how in (None, 'first'):
        return 'first({}, True)'.format(translator.translate(arg))
    elif how == 'last':
        return 'last({}, True)'.format(translator.translate(arg))
    else:
        raise com.UnsupportedOperationError(
            '{!r} value not supported for arbitrary in Spark SQL'.format(how)
        )


@compiles(ops.Strftime)
def spark_compiles_strftime(translator, expr):
    arg, format_string = expr.op().args

    return 'date_format({}, {})'.format(
        translator.translate(arg),
        translator.translate(format_string)
    )


@rewrites(ops.DayOfWeekName)
def spark_rewrites_day_of_week_name(expr):
    arg = expr.op().arg
    # TODO: if strftime gets rewritten to be compatible with ibis standard
    # strftime, then 'EEEEE' needs to be rewritten to '%A'
    return arg.strftime('EEEEE')


class SparkSelect(ImpalaSelect):
    translator = SparkExprTranslator


class SparkDialect(ImpalaDialect):
    translator = SparkExprTranslator


dialect = SparkDialect
