import ibis.sql.compiler as comp
import ibis.expr.operations as ops
from ibis.impala.compiler import ImpalaSelect
from ibis.impala import compiler as impala_compiler


class BigQuerySelectBuilder(comp.SelectBuilder):

    @property
    def _select_class(self):
        return BigQuerySelect


class BigQueryQueryBuilder(comp.QueryBuilder):

    select_builder = BigQuerySelectBuilder

    def __init__(self, expr, context=None, params=None):
        super(BigQueryQueryBuilder, self).__init__(
            expr, context=context, params=params
        )

    def _make_context(self):
        return BigQueryContext()

    @property
    def _union_class(self):
        # return BigQueryUnion
        raise NotImplementedError()


def build_ast(expr, context=None, params=None):
    builder = BigQueryQueryBuilder(expr, context=context, params=params)
    return builder.get_result()


def _get_query(expr, context, params=None):
    ast = build_ast(expr, context, params=params)
    (query, rest) = (ast.queries[0], ast.queries[1:])
    assert not rest
    return query


def to_sql(expr, context=None, params=None):
    query = _get_query(expr, context, params=params)
    compiled = query.compile()
    return compiled


class BigQueryContext(comp.QueryContext):

    def _to_sql(self, expr, ctx):
        return to_sql(expr, context=ctx)


class BigQuerySelect(ImpalaSelect):

    @property
    def translator(self):
        return BigQueryExprTranslator


def _extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.args[0])
        return 'EXTRACT({} from {})'.format(sql_attr, arg)
    return extract_field_formatter


def _ifnull(translator, expr):
    return 'IFNULL({}, {})'.format(*map(translator.translate, expr.op().args))


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


def _array_collect(translator, expr):
    arg, = expr.op().args
    return 'ARRAY_AGG({})'.format(translator.translate(arg))


def _array_concat(translator, expr):
    return 'ARRAY_CONCAT({})'.format(
        ', '.join(map(translator.translate, expr.op().args))
    )


def _array_index(translator, expr):
    # SAFE_OFFSET returns NULL if out of bounds
    return '{}[SAFE_OFFSET({})]'.format(
        *map(translator.translate, expr.op().args)
    )


def _array_length(translator, expr):
    return 'ARRAY_LENGTH({})'.format(
        *map(translator.translate, expr.op().args)
    )


def _string_replace(translator, expr):
    return 'REPLACE({}, {}, {})'.format(
        *map(translator.translate, expr.op().args)
    )


def _string_split(translator, expr):
    return 'SPLIT({}, {})'.format(
        *map(translator.translate, expr.op().args)
    )


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


_operation_registry = impala_compiler._operation_registry.copy()
_operation_registry.update({
    ops.ExtractYear: _extract_field('year'),
    ops.ExtractMonth: _extract_field('month'),
    ops.ExtractDay: _extract_field('day'),
    ops.ExtractHour: _extract_field('hour'),
    ops.ExtractMinute: _extract_field('minute'),
    ops.ExtractSecond: _extract_field('second'),
    ops.ExtractMillisecond: _extract_field('millisecond'),

    ops.StringReplace: _string_replace,
    ops.StringSplit: _string_split,
    ops.StringConcat: _string_concat,
    ops.StringJoin: _string_join,

    ops.IfNull: _ifnull,
    ops.Cast: _cast,

    ops.StructField: _struct_field,

    ops.ArrayCollect: _array_collect,
    ops.ArrayConcat: _array_concat,
    ops.ArrayIndex: _array_index,
    ops.ArrayLength: _array_length,

    # BigQuery doesn't have these operations built in.
    # ops.ArrayRepeat: _array_repeat,
    # ops.ArraySlice: _array_slice,
})


class BigQueryExprTranslator(impala_compiler.ImpalaExprTranslator):
    _registry = _operation_registry
