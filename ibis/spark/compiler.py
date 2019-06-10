import ibis
import ibis.common as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.impala.compiler as impala_compiler
import ibis.sql.compiler as comp
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


class SparkUnion(comp.Union):
    @staticmethod
    def keyword(distinct):
        return 'UNION DISTINCT' if distinct else 'UNION ALL'


class SparkQueryBuilder(comp.QueryBuilder):
    select_builder = SparkSelectBuilder
    union_class = SparkUnion


def build_ast(expr, context):
    builder = SparkQueryBuilder(expr, context=context)
    return builder.get_result()


class SparkContext(ImpalaContext):
    pass


def _array_literal_format(translator, expr):
    translated_values = [
        translator.translate(ibis.literal(x))
        for x in expr.op().value
    ]

    return 'array({})'.format(
        ', '.join(translated_values)
    )


def _literal(translator, expr):
    try:
        return impala_compiler._literal(translator, expr)
    except NotImplementedError:
        if isinstance(expr, ir.ArrayValue):
            return _array_literal_format(translator, expr)
        raise NotImplementedError(type(expr).__name__)


_operation_registry = impala_compiler._operation_registry.copy()
_operation_registry.update(
    {
        ops.IfNull: fixed_arity('ifnull', 2),
        ops.ArrayLength: unary('size'),
        ops.HLLCardinality: _reduction('approx_count_distinct'),
        ops.StrRight: fixed_arity('right', 2),
        ops.StringSplit: fixed_arity('SPLIT', 2),
        ops.RegexSearch: fixed_arity('rlike', 2),
        ops.ArrayConcat: fixed_arity('concat', 2),
        ops.Literal: _literal,
    }
)


class SparkExprTranslator(ImpalaExprTranslator):
    _registry = _operation_registry

    context_class = SparkContext


compiles = SparkExprTranslator.compiles


@compiles(ops.Arbitrary)
def spark_arbitrary(translator, expr):
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


class SparkSelect(ImpalaSelect):
    translator = SparkExprTranslator


class SparkDialect(ImpalaDialect):
    translator = SparkExprTranslator


dialect = SparkDialect
