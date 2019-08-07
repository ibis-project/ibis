import collections
import enum
import functools

import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as types
from ibis.pyspark.operations import PySparkTable


class AggregationContext(enum.Enum):
    ENTIRE = 0
    WINDOW = 1
    GROUP = 2


class PySparkExprTranslator:
    _registry = {}

    @classmethod
    def compiles(cls, klass):
        def decorator(f):
            cls._registry[klass] = f
            return f

        return decorator

    def translate(self, expr, scope, **kwargs):
        # The operation node type the typed expression wraps
        op = expr.op()

        if type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, expr, scope, **kwargs)
        else:
            raise com.OperationNotDefinedError(
                'No translation rule for {}'.format(type(op))
            )


compiles = PySparkExprTranslator.compiles


def compile_with_scope(t, expr, scope):
    """Compile a expression and put the result in scope.

       If the expression is already in scope, return it.
    """
    op = expr.op()

    if op in scope:
        result = scope[op]
    else:
        result = t.translate(expr, scope)
        scope[op] = result

    return result


@compiles(PySparkTable)
def compile_datasource(t, expr, scope):
    op = expr.op()
    name, _, client = op.args
    return client._session.table(name)


@compiles(ops.Selection)
def compile_selection(t, expr, scope, **kwargs):
    # Cache compile results for tables
    op = expr.op()

    # TODO: Support predicates and sort_keys
    if op.predicates or op.sort_keys:
        raise NotImplementedError(
            "predicates and sort_keys are not supported with Selection")

    src_table = compile_with_scope(t, op.table, scope)
    col_names_in_selection_order = []

    for selection in op.selections:
        if isinstance(selection, types.TableExpr):
            col_names_in_selection_order.extend(selection.columns)
        elif isinstance(selection, types.ColumnExpr):
            column_name = selection.get_name()
            col_names_in_selection_order.append(column_name)
            column = t.translate(selection, scope=scope)
            src_table = src_table.withColumn(column_name, column)

    return src_table[col_names_in_selection_order]


@compiles(ops.TableColumn)
def compile_column(t, expr, scope, **kwargs):
    op = expr.op()
    table = compile_with_scope(t, op.table, scope)
    return table[op.name]


@compiles(ops.DistinctColumn)
def compile_distinct(t, expr, scope, **kwargs):
    op = expr.op()
    root_table_expr = op.root_tables()[0].to_expr()
    src_table = compile_with_scope(t, root_table_expr, scope)
    src_column_name = op.arg.get_name()
    return src_table.select(src_column_name).distinct()[src_column_name]


@compiles(ops.SelfReference)
def compile_self_reference(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.table, scope)


@compiles(ops.Equals)
def compile_equals(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) == t.translate(op.right, scope)


@compiles(ops.Greater)
def compile_greater(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) > t.translate(op.right, scope)


@compiles(ops.GreaterEqual)
def compile_greater_equal(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) >= t.translate(op.right, scope)


@compiles(ops.Multiply)
def compile_multiply(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) * t.translate(op.right, scope)


@compiles(ops.Subtract)
def compile_subtract(t, expr, scope, **kwargs):
    op = expr.op()
    return t.translate(op.left, scope) - t.translate(op.right, scope)


@compiles(ops.Literal)
def compile_literal(t, expr, scope, raw=False, **kwargs):
    """ If raw is True, don't wrap the result with F.lit()
    """
    value = expr.op().value

    if raw:
        return value

    if isinstance(value, collections.abc.Set):
        # Don't wrap set with F.lit
        if isinstance(value, frozenset):
            # Spark doens't like frozenset
            return set(value)
        else:
            return value
    else:
        return F.lit(expr.op().value)


@compiles(ops.Aggregation)
def compile_aggregation(t, expr, scope, **kwargs):
    op = expr.op()

    src_table = t.translate(op.table, scope)

    if op.by:
        context = AggregationContext.GROUP
        aggs = [t.translate(m, scope, context=context)
                for m in op.metrics]
        bys = [t.translate(b, scope) for b in op.by]
        return src_table.groupby(*bys).agg(*aggs)
    else:
        context = AggregationContext.ENTIRE
        aggs = [t.translate(m, scope, context=context)
                for m in op.metrics]
        return src_table.agg(*aggs)


@compiles(ops.Contains)
def compile_contains(t, expr, scope, **kwargs):
    op = expr.op()
    col = t.translate(op.value, scope)
    return col.isin(t.translate(op.options, scope))


def compile_aggregator(t, expr, scope, fn, context=None, **kwargs):
    op = expr.op()
    src_col = t.translate(op.arg, scope)

    if getattr(op, "where", None) is not None:
        condition = t.translate(op.where, scope)
        src_col = F.when(condition, src_col)

    col = fn(src_col)
    if context:
        return col
    else:
        # We are trying to compile a expr such as some_col.max()
        # to a Spark expression.
        # Here we get the root table df of that column and compile
        # the expr to:
        # df.select(max(some_col))
        return t.translate(expr.op().arg.op().table, scope).select(col)


@compiles(ops.GroupConcat)
def compile_group_concat(t, expr, scope, context=None, **kwargs):
    sep = expr.op().sep.op().value

    def fn(col):
        return F.concat_ws(sep, F.collect_list(col))
    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Any)
def compile_any(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.max, context)


@compiles(ops.NotAny)
def compile_notany(t, expr, scope, context=None, **kwargs):

    def fn(col):
        return ~F.max(col)
    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.All)
def compile_all(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.min, context)


@compiles(ops.NotAll)
def compile_notall(t, expr, scope, context=None, **kwargs):

    def fn(col):
        return ~F.min(col)
    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Count)
def compile_count(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.count, context)


@compiles(ops.Max)
def compile_max(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.max, context)


@compiles(ops.Min)
def compile_min(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.min, context)


@compiles(ops.Mean)
def compile_mean(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.mean, context)


@compiles(ops.Sum)
def compile_sum(t, expr, scope, context=None, **kwargs):
    return compile_aggregator(t, expr, scope, F.sum, context)


@compiles(ops.StandardDev)
def compile_std(t, expr, scope, context=None, **kwargs):
    how = expr.op().how

    if how == 'sample':
        fn = F.stddev_samp
    elif how == 'pop':
        fn = F.stddev_pop
    else:
        raise com.TranslationError(
            "Unexpected 'how' in translation: {}"
            .format(how)
        )

    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Variance)
def compile_variance(t, expr, scope, context=None, **kwargs):
    how = expr.op().how

    if how == 'sample':
        fn = F.var_samp
    elif how == 'pop':
        fn = F.var_pop
    else:
        raise com.TranslationError(
            "Unexpected 'how' in translation: {}"
            .format(how)
        )

    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Arbitrary)
def compile_arbitrary(t, expr, scope, context=None, **kwargs):
    how = expr.op().how

    if how == 'first':
        fn = functools.partial(F.first, ignorenulls=True)
    elif how == 'last':
        fn = functools.partial(F.last, ignorenulls=True)
    else:
        raise NotImplementedError(
            "Does not support 'how': {}".format(how)
        )

    return compile_aggregator(t, expr, scope, fn, context)


@compiles(ops.Greatest)
def compile_greatest(t, expr, scope, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.greatest(*src_columns)


@compiles(ops.Least)
def compile_least(t, expr, scope, **kwargs):
    op = expr.op()

    src_columns = t.translate(op.arg, scope)
    if len(src_columns) == 1:
        return src_columns[0]
    else:
        return F.least(*src_columns)


@compiles(ops.Abs)
def compile_abs(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.abs(src_column)


@compiles(ops.Round)
def compile_round(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    scale = (t.translate(op.digits, scope, raw=True)
             if op.digits is not None else 0)
    rounded = F.round(src_column, scale=scale)
    if scale == 0:
        rounded = rounded.astype('long')
    return rounded


@compiles(ops.Ceil)
def compile_ceil(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.ceil(src_column)


@compiles(ops.Floor)
def compile_floor(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.floor(src_column)


@compiles(ops.Exp)
def compile_exp(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.exp(src_column)


@compiles(ops.Sign)
def compile_sign(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)

    return F.when(src_column == 0, F.lit(0.0)) \
        .otherwise(F.when(src_column > 0, F.lit(1.0)).otherwise(-1.0))


@compiles(ops.Sqrt)
def compile_sqrt(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.sqrt(src_column)


@compiles(ops.Log)
def compile_log(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    # Spark log method only takes float
    return F.log(float(t.translate(op.base, scope, raw=True)), src_column)


@compiles(ops.Ln)
def compile_ln(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.log(src_column)


@compiles(ops.Log2)
def compile_log2(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.log2(src_column)


@compiles(ops.Log10)
def compile_log10(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.log10(src_column)


@compiles(ops.Modulus)
def compile_modulus(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return left % right


@compiles(ops.Negate)
def compile_negate(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return -src_column


@compiles(ops.Add)
def compile_add(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return left + right


@compiles(ops.Divide)
def compile_divide(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return left / right


@compiles(ops.FloorDivide)
def compile_floor_divide(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return F.floor(left / right)


@compiles(ops.Power)
def compile_power(t, expr, scope, **kwargs):
    op = expr.op()

    left = t.translate(op.left, scope)
    right = t.translate(op.right, scope)
    return F.pow(left, right)


@compiles(ops.IsNan)
def compile_isnan(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.isnan(src_column)


@compiles(ops.IsInf)
def compile_isinf(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return (src_column == float('inf')) | (src_column == float('-inf'))


@compiles(ops.Uppercase)
def compile_uppercase(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def upper(v):
        return v.str.upper()

    src_column = t.translate(op.arg, scope)
    return upper(src_column)


@compiles(ops.Lowercase)
def compile_lowercase(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def lower(v):
        return v.str.lower()

    src_column = t.translate(op.arg, scope)
    return lower(src_column)


@compiles(ops.Reverse)
def compile_reverse(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def reverse(s):
        return s.str[::-1]

    src_column = t.translate(op.arg, scope)
    return reverse(src_column)


@compiles(ops.Strip)
def compile_strip(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def strip(s):
        return s.str.strip()

    src_column = t.translate(op.arg, scope)
    return strip(src_column)


@compiles(ops.LStrip)
def compile_lstrip(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def lstrip(s):
        return s.str.lstrip()

    src_column = t.translate(op.arg, scope)
    return lstrip(src_column)


@compiles(ops.RStrip)
def compile_rstrip(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def rstrip(s):
        return s.str.lstrip()

    src_column = t.translate(op.arg, scope)
    return rstrip(src_column)


@compiles(ops.Capitalize)
def compile_capitalize(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('string', PandasUDFType.SCALAR)
    def capitalize(s):
        return s.str.capitalize()

    src_column = t.translate(op.arg, scope)
    return capitalize(src_column)


@compiles(ops.Substring)
def compile_substring(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('string')
    def substring(s, start, length):
        end = start + length
        return s[start:end]

    src_column = t.translate(op.arg, scope)
    start_column = t.translate(op.start, scope)
    length_column = t.translate(op.length, scope)
    return substring(src_column, start_column, length_column)


@compiles(ops.StringLength)
def compile_string_length(t, expr, scope, **kwargs):
    op = expr.op()

    @pandas_udf('int', PandasUDFType.SCALAR)
    def length(s):
        return s.str.len()

    src_column = t.translate(op.arg, scope)
    return length(src_column)


@compiles(ops.StrRight)
def compile_str_right(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('string')
    def str_right(s, nchars):
        return s[-nchars:]

    src_column = t.translate(op.arg, scope)
    nchars_column = t.translate(op.nchars, scope)
    return str_right(src_column, nchars_column)


@compiles(ops.Repeat)
def compile_repeat(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('string')
    def repeat(s, times):
        return s * times

    src_column = t.translate(op.arg, scope)
    times_column = t.translate(op.times, scope)
    return repeat(src_column, times_column)


@compiles(ops.StringFind)
def compile_string_find(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('long')
    def str_find(s, substr, start, end):
        return s.find(substr, start, end)

    src_column = t.translate(op.arg, scope)
    substr_column = t.translate(op.substr, scope)
    start_column = t.translate(op.start, scope) if op.start else F.lit(None)
    end_column = t.translate(op.end, scope) if op.end else F.lit(None)
    return str_find(src_column, substr_column, start_column, end_column)


@compiles(ops.Translate)
def compile_translate(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    from_str = op.from_str.op().value
    to_str = op.to_str.op().value
    return F.translate(src_column, from_str, to_str)


@compiles(ops.LPad)
def compile_lpad(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    length = op.length.op().value
    pad = op.pad.op().value
    return F.lpad(src_column, length, pad)


@compiles(ops.RPad)
def compile_rpad(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    length = op.length.op().value
    pad = op.pad.op().value
    return F.rpad(src_column, length, pad)


@compiles(ops.StringJoin)
def compile_string_join(t, expr, scope, **kwargs):
    op = expr.op()

    @F.udf('string')
    def join(sep, arr):
        return sep.join(arr)

    sep_column = t.translate(op.sep, scope)
    arg = t.translate(op.arg, scope)
    return join(sep_column, F.array(arg))


@compiles(ops.RegexSearch)
def compile_regex_search(t, expr, scope, **kwargs):
    import re
    op = expr.op()

    @F.udf('boolean')
    def regex_search(s, pattern):
        return True if re.search(pattern, s) else False

    src_column = t.translate(op.arg, scope)
    pattern = t.translate(op.pattern, scope)
    return regex_search(src_column, pattern)


@compiles(ops.RegexExtract)
def compile_regex_extract(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    pattern = op.pattern.op().value
    idx = op.index.op().value
    return F.regexp_extract(src_column, pattern, idx)


@compiles(ops.RegexReplace)
def compile_regex_replace(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    pattern = op.pattern.op().value
    replacement = op.replacement.op().value
    return F.regexp_replace(src_column, pattern, replacement)


@compiles(ops.StringReplace)
def compile_string_replace(t, expr, scope, **kwargs):
    return compile_regex_replace(t, expr)


@compiles(ops.StringSplit)
def compile_string_split(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    delimiter = op.delimiter.op().value
    return F.split(src_column, delimiter)


@compiles(ops.StringAscii)
def compile_string_ascii(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    return F.ascii(src_column)


@compiles(ops.StringSQLLike)
def compile_string_like(t, expr, scope, **kwargs):
    op = expr.op()

    src_column = t.translate(op.arg, scope)
    pattern = op.pattern.op().value
    return src_column.like(pattern)


@compiles(ops.ValueList)
def compile_value_list(t, expr, scope, **kwargs):
    op = expr.op()
    return [t.translate(col, scope) for col in op.values]


@compiles(ops.InnerJoin)
def compile_inner_join(t, expr, scope, **kwargs):
    return compile_join(t, expr, scope, 'inner')


def compile_join(t, expr, scope, how):
    op = expr.op()

    left_df = t.translate(op.left, scope)
    right_df = t.translate(op.right, scope)
    # TODO: Handle multiple predicates
    predicates = t.translate(op.predicates[0], scope)

    return left_df.join(right_df, predicates, how)
