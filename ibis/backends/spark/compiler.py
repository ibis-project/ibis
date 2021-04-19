"""
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

"""


import math

import ibis
import ibis.backends.base_sqlalchemy.compiler as comp
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.base_sql.compiler import (
    BaseDialect,
    BaseExprTranslator,
    BaseSelect,
)

from .registry import operation_registry


# ----------------------------------------------------------------------
# Select compilation


class SparkUDFNode(ops.ValueOp):
    def output_type(self):
        return rlz.shape_like(self.args, dtype=self.return_type)


class SparkUDAFNode(ops.Reduction):
    def output_type(self):
        return self.return_type.scalar_type()


class SparkSelectBuilder(comp.SelectBuilder):
    @property
    def _select_class(self):
        return SparkSelect


class SparkQueryBuilder(comp.QueryBuilder):
    select_builder = SparkSelectBuilder


class SparkContext(comp.QueryContext):
    def _to_sql(self, expr, ctx):
        if ctx is None:
            ctx = BaseDialect.make_context()
        builder = SparkQueryBuilder(expr, context=ctx)
        ast = builder.get_result()
        query = ast.queries[0]
        return query.compile()


class SparkExprTranslator(BaseExprTranslator):
    _registry = operation_registry

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


@compiles(ops.DayOfWeekName)
def spark_compiles_day_of_week_name(translator, expr):
    arg = expr.op().arg
    return 'date_format({}, {!r})'.format(translator.translate(arg), 'EEEE')


@rewrites(ops.IsInf)
def spark_rewrites_is_inf(expr):
    arg = expr.op().arg
    return (arg == ibis.literal(math.inf)) | (arg == ibis.literal(-math.inf))


class SparkSelect(BaseSelect):
    translator = SparkExprTranslator
