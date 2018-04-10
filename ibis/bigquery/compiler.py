from functools import partial

import six

from multipledispatch import Dispatcher

import ibis
import ibis.common as com

import numpy as np

import ibis.expr.datatypes as dt
import ibis.expr.types as ir

import ibis.sql.compiler as comp
import ibis.expr.operations as ops
import ibis.expr.lineage as lin

from ibis.impala.compiler import ImpalaSelect, unary, fixed_arity
from ibis.impala import compiler as impala_compiler

from ibis.bigquery.types import ibis_type_to_bigquery_type


class BigQueryUDFNode(ops.ValueOp):
    pass


class BigQuerySelectBuilder(comp.SelectBuilder):

    @property
    def _select_class(self):
        return BigQuerySelect


class BigQueryUDFDefinition(comp.DDL):

    def __init__(self, expr, context):
        self.expr = expr
        self.context = context

    def compile(self):
        return self.expr.op().js


class BigQueryUnion(comp.Union):
    @property
    def keyword(self):
        return 'UNION DISTINCT' if self.distinct else 'UNION ALL'


def find_bigquery_udf(expr):
    if isinstance(expr.op(), BigQueryUDFNode):
        result = expr
    else:
        result = None
    return lin.proceed, result


class BigQueryQueryBuilder(comp.QueryBuilder):

    select_builder = BigQuerySelectBuilder
    union_class = BigQueryUnion

    def generate_setup_queries(self):
        result = list(
            map(partial(BigQueryUDFDefinition, context=self.context),
                lin.traverse(find_bigquery_udf, self.expr)))
        return result


def build_ast(expr, context):
    builder = BigQueryQueryBuilder(expr, context=context)
    return builder.get_result()


def to_sql(expr, context):
    query_ast = build_ast(expr, context)
    compiled = query_ast.compile()
    return compiled


class BigQueryContext(comp.QueryContext):

    def _to_sql(self, expr, ctx):
        return to_sql(expr, context=ctx)


def _extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.args[0])
        return 'EXTRACT({} from {})'.format(sql_attr, arg)
    return extract_field_formatter


SQL_TYPE_NAMES = {
    'int8': 'int64',
    'int16': 'int64',
    'int32': 'int64',
    'int64': 'int64',
    'float': 'float64',
    'double': 'float64',
    'string': 'string',
    'boolean': 'boolean',
    'timestamp': 'timestamp',
    'date': 'date',
}


bigquery_cast = Dispatcher('bigquery_cast')


@bigquery_cast.register(six.string_types, dt.Timestamp, dt.Integer)
def bigquery_cast_timestamp_to_integer(compiled_arg, from_, to):
    return 'UNIX_MICROS({})'.format(compiled_arg)


@bigquery_cast.register(six.string_types, dt.DataType, dt.DataType)
def bigquery_cast_generate(compiled_arg, from_, to):
    target_name = to.name.lower()
    sql_type = SQL_TYPE_NAMES[target_name]
    uppercase_sql_type = sql_type.upper()
    return 'CAST({} AS {})'.format(compiled_arg, uppercase_sql_type)


def _cast(translator, expr):
    op = expr.op()
    arg, target_type = op.args
    arg_formatted = translator.translate(arg)
    return bigquery_cast(arg_formatted, arg.type(), target_type)


def _struct_field(translator, expr):
    arg, field = expr.op().args
    arg_formatted = translator.translate(arg)
    return '{}.`{}`'.format(arg_formatted, field)


def _array_concat(translator, expr):
    return 'ARRAY_CONCAT({})'.format(
        ', '.join(map(translator.translate, expr.op().args))
    )


def _array_index(translator, expr):
    # SAFE_OFFSET returns NULL if out of bounds
    return '{}[SAFE_OFFSET({})]'.format(
        *map(translator.translate, expr.op().args)
    )


def _string_find(translator, expr):
    haystack, needle, start, end = expr.op().args

    if start is not None:
        raise NotImplementedError('start not implemented for string find')
    if end is not None:
        raise NotImplementedError('end not implemented for string find')

    return 'STRPOS({}, {}) - 1'.format(
        translator.translate(haystack),
        translator.translate(needle)
    )


def _translate_pattern(translator, pattern):
    # add 'r' to string literals to indicate to BigQuery this is a raw string
    return 'r' * isinstance(pattern.op(), ops.Literal) + translator.translate(
        pattern
    )


def _regex_search(translator, expr):
    arg, pattern = expr.op().args
    regex = _translate_pattern(translator, pattern)
    result = 'REGEXP_CONTAINS({}, {})'.format(translator.translate(arg), regex)
    return result


def _regex_extract(translator, expr):
    arg, pattern, index = expr.op().args
    regex = _translate_pattern(translator, pattern)
    result = 'REGEXP_EXTRACT_ALL({}, {})[SAFE_OFFSET({})]'.format(
        translator.translate(arg),
        regex,
        translator.translate(index)
    )
    return result


def _regex_replace(translator, expr):
    arg, pattern, replacement = expr.op().args
    regex = _translate_pattern(translator, pattern)
    result = 'REGEXP_REPLACE({}, {}, {})'.format(
        translator.translate(arg),
        regex,
        translator.translate(replacement),
    )
    return result


def _string_concat(translator, expr):
    return 'CONCAT({})'.format(
        ', '.join(map(translator.translate, expr.op().arg))
    )


def _string_join(translator, expr):
    sep, args = expr.op().args
    return 'ARRAY_TO_STRING([{}], {})'.format(
        ', '.join(map(translator.translate, args)),
        translator.translate(sep)
    )


def _string_ascii(translator, expr):
    arg, = expr.op().args
    return 'TO_CODE_POINTS({})[SAFE_OFFSET(0)]'.format(
        translator.translate(arg)
    )


def _string_right(translator, expr):
    arg, nchars = map(translator.translate, expr.op().args)
    return 'SUBSTR({arg}, -LEAST(LENGTH({arg}), {nchars}))'.format(
        arg=arg,
        nchars=nchars,
    )


def _array_literal_format(expr):
    return str(list(expr.op().value))


def _log(translator, expr):
    op = expr.op()
    arg, base = op.args
    arg_formatted = translator.translate(arg)

    if base is None:
        return 'ln({})'.format(arg_formatted)

    base_formatted = translator.translate(base)
    return 'log({}, {})'.format(arg_formatted, base_formatted)


def _literal(translator, expr):

    if isinstance(expr, ir.NumericValue):
        value = expr.op().value
        if not np.isfinite(value):
            return 'CAST({!r} AS FLOAT64)'.format(str(value))

    try:
        return impala_compiler._literal(translator, expr)
    except NotImplementedError:
        if isinstance(expr, ir.ArrayValue):
            return _array_literal_format(expr)
        raise NotImplementedError(type(expr).__name__)


def _arbitrary(translator, expr):
    arg, how, where = expr.op().args

    if where is not None:
        arg = where.ifelse(arg, ibis.NA)

    if how != 'first':
        raise com.UnsupportedOperationError(
            '{!r} value not supported for arbitrary in BigQuery'.format(how)
        )

    return 'ANY_VALUE({})'.format(translator.translate(arg))


_date_units = {
    'Y': 'YEAR',
    'Q': 'QUARTER',
    'W': 'WEEK',
    'M': 'MONTH',
}


_timestamp_units = {
    'us': 'MICROSECOND',
    'ms': 'MILLISECOND',
    's': 'SECOND',
    'm': 'MINUTE',
    'h': 'HOUR',
}
_timestamp_units.update(_date_units)


def _truncate(kind, units):
    def truncator(translator, expr):
        op = expr.op()
        arg, unit = op.args

        arg = translator.translate(op.args[0])
        try:
            unit = units[unit]
        except KeyError:
            raise com.UnsupportedOperationError(
                '{!r} unit is not supported in timestamp truncate'.format(unit)
            )

        return "{}_TRUNC({}, {})".format(kind, arg, unit)
    return truncator


def _timestamp_op(func, units):
    def _formatter(translator, expr):
        op = expr.op()
        arg, offset = op.args

        unit = offset.type().unit
        if unit not in units:
            raise com.UnsupportedOperationError(
                'BigQuery does not allow binary operation '
                '{} with INTERVAL offset {}'.format(func, unit)
            )
        formatted_arg = translator.translate(arg)
        formatted_offset = translator.translate(offset)
        result = '{}({}, {})'.format(func, formatted_arg, formatted_offset)
        return result

    return _formatter


STRFTIME_FORMAT_FUNCTIONS = {
    dt.Date: 'DATE',
    dt.Time: 'TIME',
    dt.Timestamp: 'TIMESTAMP',
}


_operation_registry = impala_compiler._operation_registry.copy()
_operation_registry.update({
    ops.ExtractYear: _extract_field('year'),
    ops.ExtractMonth: _extract_field('month'),
    ops.ExtractDay: _extract_field('day'),
    ops.ExtractHour: _extract_field('hour'),
    ops.ExtractMinute: _extract_field('minute'),
    ops.ExtractSecond: _extract_field('second'),
    ops.ExtractMillisecond: _extract_field('millisecond'),

    ops.StringReplace: fixed_arity('REPLACE', 3),
    ops.StringSplit: fixed_arity('SPLIT', 2),
    ops.StringConcat: _string_concat,
    ops.StringJoin: _string_join,
    ops.StringAscii: _string_ascii,
    ops.StringFind: _string_find,
    ops.StrRight: _string_right,
    ops.Repeat: fixed_arity('REPEAT', 2),
    ops.RegexSearch: _regex_search,
    ops.RegexExtract: _regex_extract,
    ops.RegexReplace: _regex_replace,

    ops.GroupConcat: fixed_arity('STRING_AGG', 2),

    ops.IfNull: fixed_arity('IFNULL', 2),
    ops.Cast: _cast,

    ops.StructField: _struct_field,

    ops.ArrayCollect: unary('ARRAY_AGG'),
    ops.ArrayConcat: _array_concat,
    ops.ArrayIndex: _array_index,
    ops.ArrayLength: unary('ARRAY_LENGTH'),

    ops.Log: _log,
    ops.Modulus: fixed_arity('MOD', 2),

    ops.Date: unary('DATE'),

    # BigQuery doesn't have these operations built in.
    # ops.ArrayRepeat: _array_repeat,
    # ops.ArraySlice: _array_slice,
    ops.Literal: _literal,
    ops.Arbitrary: _arbitrary,

    ops.TimestampTruncate: _truncate('TIMESTAMP', _timestamp_units),
    ops.DateTruncate: _truncate('DATE', _date_units),

    ops.TimestampAdd: _timestamp_op(
        'TIMESTAMP_ADD', {'h', 'm', 's', 'ms', 'us'}),
    ops.TimestampSub: _timestamp_op(
        'TIMESTAMP_DIFF', {'h', 'm', 's', 'ms', 'us'}),

    ops.DateAdd: _timestamp_op('DATE_ADD', {'D', 'W', 'M', 'Q', 'Y'}),
    ops.DateSub: _timestamp_op('DATE_SUB', {'D', 'W', 'M', 'Q', 'Y'}),
})

_invalid_operations = {
    ops.Translate,
    ops.FindInSet,
    ops.Capitalize,
    ops.DateDiff,
    ops.TimestampDiff
}

_operation_registry = {
    k: v for k, v in _operation_registry.items()
    if k not in _invalid_operations
}


class BigQueryExprTranslator(impala_compiler.ImpalaExprTranslator):
    _registry = _operation_registry
    _rewrites = impala_compiler.ImpalaExprTranslator._rewrites.copy()

    context_class = BigQueryContext

    def _trans_param(self, expr):
        op = expr.op()
        if op not in self.context.params:
            raise KeyError(op)
        return '@{}'.format(expr.get_name())


compiles = BigQueryExprTranslator.compiles
rewrites = BigQueryExprTranslator.rewrites


@compiles(ops.Divide)
def bigquery_compiles_divide(t, e):
    return 'IEEE_DIVIDE({}, {})'.format(*map(t.translate, e.op().args))


@compiles(ops.Strftime)
def compiles_strftime(translator, expr):
    arg, format_string = expr.op().args
    arg_type = arg.type()
    strftime_format_func_name = STRFTIME_FORMAT_FUNCTIONS[type(arg_type)]
    fmt_string = translator.translate(format_string)
    arg_formatted = translator.translate(arg)
    if isinstance(arg_type, dt.Timestamp):
        return 'FORMAT_{}({}, {}, {!r})'.format(
            strftime_format_func_name,
            fmt_string,
            arg_formatted,
            arg_type.timezone if arg_type.timezone is not None else 'UTC'
        )
    return 'FORMAT_{}({}, {})'.format(
        strftime_format_func_name,
        fmt_string,
        arg_formatted
    )


@rewrites(ops.Any)
def bigquery_rewrite_any(expr):
    arg, = expr.op().args
    return arg.cast(dt.int64).sum() > 0


@rewrites(ops.NotAny)
def bigquery_rewrite_notany(expr):
    arg, = expr.op().args
    return arg.cast(dt.int64).sum() == 0


@rewrites(ops.All)
def bigquery_rewrite_all(expr):
    arg, = expr.op().args
    return (1 - arg.cast(dt.int64)).sum() == 0


@rewrites(ops.NotAll)
def bigquery_rewrite_notall(expr):
    arg, = expr.op().args
    return (1 - arg.cast(dt.int64)).sum() != 0


class BigQuerySelect(ImpalaSelect):

    translator = BigQueryExprTranslator


@rewrites(ops.IdenticalTo)
def identical_to(expr):
    left, right = expr.op().args
    return (left.isnull() & right.isnull()) | (left == right)


@rewrites(ops.Log2)
def log2(expr):
    arg, = expr.op().args
    return arg.log(2)


UNIT_FUNCS = {
    's': 'SECONDS',
    'ms': 'MILLIS',
    'us': 'MICROS',
}


@compiles(ops.TimestampFromUNIX)
def compiles_timestamp_from_unix(t, e):
    value, unit = e.op().args
    return 'TIMESTAMP_{}({})'.format(UNIT_FUNCS[unit], t.translate(value))


@compiles(ops.Floor)
def compiles_floor(t, e):
    bigquery_type = ibis_type_to_bigquery_type(e.type())
    arg, = e.op().args
    return 'CAST(FLOOR({}) AS {})'.format(t.translate(arg), bigquery_type)


class BigQueryDialect(impala_compiler.ImpalaDialect):

    translator = BigQueryExprTranslator


dialect = BigQueryDialect
