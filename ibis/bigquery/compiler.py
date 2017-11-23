import ibis
import ibis.sql.compiler as comp
import ibis.expr.operations as ops
from ibis.bigquery import operations as bq_ops
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
        return "extract({0!s} from {1!s})".format(sql_attr, arg)
    return extract_field_formatter


def _ifnull(translator, expr):
    (a, b) = (translator.translate(arg) for arg in expr.op().args)
    return ('CASE WHEN {0!s} IS NULL THEN {1!s} ELSE {0!s} END'
            .format(a, b))


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
    return 'CAST({0!s} AS {1!s})'.format(arg_formatted, sql_type)


def _struct_field(translator, expr):
    arg, field = expr.op().args
    arg_formatted = translator.translate(arg)
    return '{}.`{}`'.format(arg_formatted, field)


def _array_collect(translator, expr):
    return 'ARRAY_AGG({})'.format(*map(translator.translate, expr.op().args))


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


def _arbitrary(translator, expr):
    arg, where = expr.op().args
    if where is not None:
        arg_formatted = translator.translate(where.ifelse(arg, ibis.NA))
    else:
        arg_formatted = translator.translate(arg)
    return 'ANY_VALUE({})'.format(arg_formatted)


def _percentile(translator, expr):
    args = expr.op().args
    (arg, percentile_rank) = map(translator.translate, args[:2])
    value_type = args[2]
    command = dict(cont='percentile_cont', disc='percentile_disc')[value_type]
    return '{}({}, {})'.format(command, arg, percentile_rank)


def _approx_quantile(translator, expr):
    args = expr.op().args
    args = [translator.translate(arg) for arg in args[:2]]
    # distinct?
    return 'APPROX_QUANTILES({}, {})'.format(*args)


def approx_quantile(arg, number, distinct=False, ignore_nulls=True):
    if distinct or not ignore_nulls:
        raise NotImplementedError()

    return bq_ops.ApproxQuantile(arg, number, distinct, ignore_nulls).to_expr()


def _approx_nunique(translator, expr):
    arg = expr.op().args[0]
    arg_formatted = translator.translate(arg)
    return 'APPROX_COUNT_DISTINCT({})'.format(arg_formatted)


def _format_date(translator, expr):
    (arg, fmt) = map(translator.translate, expr.op().args)
    return 'FORMAT_DATE({}, {})'.format(fmt, arg)


def format_date(fmt, arg):
    return bq_ops.FormatDate(arg, fmt).to_expr()


def _date_diff(translator, expr):
    (arg0, arg1, date_part) = expr.op().args
    (af0, af1) = [translator.translate(arg) for arg in (arg0, arg1)]
    return 'DATE_DIFF({}, {}, {})'.format(af0, af1, date_part)


def date_diff(arg0, arg1, date_part='DAY'):
    return bq_ops.DateDiff(arg0, arg1, date_part).to_expr()


def _timestamp_diff(translator, expr):
    (arg0, arg1, timestamp_part) = expr.op().args
    (af0, af1) = [translator.translate(arg) for arg in (arg0, arg1)]
    return 'TIMESTAMP_DIFF({}, {}, {})'.format(af0, af1, timestamp_part)


def timestamp_diff(arg0, arg1, timestamp_part='SECOND'):
    return bq_ops.TimestampDiff(arg0, arg1, timestamp_part).to_expr()


_operation_registry = impala_compiler._operation_registry.copy()
_operation_registry.update({
    ops.ExtractYear: _extract_field('year'),
    ops.ExtractMonth: _extract_field('month'),
    ops.ExtractDay: _extract_field('day'),
    ops.ExtractHour: _extract_field('hour'),
    ops.ExtractMinute: _extract_field('minute'),
    ops.ExtractSecond: _extract_field('second'),
    ops.ExtractMillisecond: _extract_field('millisecond'),

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
    ops.Arbitrary: _arbitrary,
    ops.Percentile: _percentile,
    bq_ops.ApproxQuantile: _approx_quantile,
    ops.HLLCardinality: _approx_nunique,
    bq_ops.DateDiff: _date_diff,
    bq_ops.TimestampDiff: _timestamp_diff,
    bq_ops.FormatDate: _format_date,
})


class BigQueryExprTranslator(impala_compiler.ImpalaExprTranslator):
    _registry = _operation_registry


def percentile_aggregation(expr, cols, percentile_ranks, names,
                           value_type='cont', group_by=()):

    assert all(isinstance(el, (list, tuple))
               for el in (cols, percentile_ranks, names))
    assert len(cols) == len(percentile_ranks) and len(cols) == len(names)
    group_by = group_by or []

    # zipped = list(zip(names, cols, percentile_ranks))
    # expr = (table
    zipped = zip(names, cols, percentile_ranks)
    expr = (expr
            .groupby(group_by)
            .mutate(**{
                name: lambda t, c=c, p=p: t[c].percentile(p, value_type)
                for (name, c, p) in zipped
                })
            .groupby(group_by)
            .aggregate(**{name: lambda t, name=name: t[name].arbitrary()
                          for name in names})
            )
    return expr
