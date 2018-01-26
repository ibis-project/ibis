import ibis.expr.datatypes as dt
import ibis.expr.types as ir

import ibis.sql.compiler as comp
import ibis.expr.operations as ops
from ibis.impala.compiler import ImpalaSelect, unary, fixed_arity
from ibis.impala import compiler as impala_compiler


class BigQuerySelectBuilder(comp.SelectBuilder):

    @property
    def _select_class(self):
        return BigQuerySelect


class BigQueryQueryBuilder(comp.QueryBuilder):

    select_builder = BigQuerySelectBuilder

    @property
    def _union_class(self):
        # return BigQueryUnion
        raise NotImplementedError()


def build_ast(expr, context):
    builder = BigQueryQueryBuilder(expr, context=context)
    return builder.get_result()


def _get_query(expr, context):
    ast = build_ast(expr, context)
    (query, rest) = (ast.queries[0], ast.queries[1:])
    assert not rest
    return query


def to_sql(expr, context):
    query = _get_query(expr, context)
    compiled = query.compile()
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


_sql_type_names = {
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


def _cast(translator, expr):
    op = expr.op()
    arg, target_type = op.args
    arg_formatted = translator.translate(arg)
    sql_type = _sql_type_names[target_type.name.lower()]
    return 'CAST({} AS {})'.format(arg_formatted, sql_type.upper())


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
    return 'r' * isinstance(pattern.op(), ir.Literal) + translator.translate(
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
        ', '.join(map(translator.translate, expr.op().args))
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


def _literal(translator, expr):
    try:
        return impala_compiler._literal(translator, expr)
    except NotImplementedError:
        if isinstance(expr, ir.ArrayValue):
            return _array_literal_format(expr)
        raise NotImplementedError(type(expr).__name__)


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

    # BigQuery doesn't have these operations built in.
    # ops.ArrayRepeat: _array_repeat,
    # ops.ArraySlice: _array_slice,
    ir.Literal: _literal,
})

_invalid_operations = {
    ops.Translate,
    ops.FindInSet,
    ops.Capitalize,
    ops.IsNan,
    ops.IsInf,
    ops.Log,
    ops.Log2,
    ops.Modulus,
    ops.Date,
    ops.IntervalFromInteger,
    ops.ExtractYear,
    ops.ExtractMonth,
    ops.ExtractDay,
    ops.ExtractHour,
    ops.ExtractMinute,
    ops.ExtractSecond,
    ops.ExtractMillisecond,
    ops.TimestampAdd,
    ops.TimestampSubtract,
    ops.DateTruncate,
    ops.TimestampTruncate,
    ops.IdenticalTo,
    ops.StringAscii,
    ops.StringLength
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
        if expr not in self.context.params:
            raise KeyError(expr)
        return '@{}'.format(expr._name)


rewrites = BigQueryExprTranslator.rewrites


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


class BigQueryDialect(impala_compiler.ImpalaDialect):

    translator = BigQueryExprTranslator


dialect = BigQueryDialect
