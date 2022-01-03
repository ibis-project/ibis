import functools
import operator

import datafusion as df
import datafusion.functions
import pyarrow as pa

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir

from .datatypes import to_pyarrow_type


@functools.singledispatch
def translate(expr):
    raise NotImplementedError(expr)


@translate.register(ir.Expr)
def expression(expr):
    return translate(expr.op(), expr)


@translate.register(ops.Node)
def operation(op, expr):
    raise com.OperationNotDefinedError(f'No translation rule for {type(op)}')


@translate.register(ops.DatabaseTable)
def table(op, expr):
    name, _, client = op.args
    return client._context.table(name)


@translate.register(ops.Literal)
def literal(op, expr):
    if isinstance(op.value, (set, frozenset)):
        value = list(op.value)
    else:
        value = op.value

    arrow_type = to_pyarrow_type(op.dtype)
    arrow_scalar = pa.scalar(value, type=arrow_type)

    return df.literal(arrow_scalar)


@translate.register(ops.Cast)
def cast(op, expr):
    arg = translate(op.arg)
    typ = to_pyarrow_type(op.to)
    return arg.cast(to=typ)


@translate.register(ops.TableColumn)
def column(op, expr):
    table_op = op.table.op()

    if hasattr(table_op, "name"):
        return df.column(f'{table_op.name}."{op.name}"')
    else:
        return df.column(op.name)


@translate.register(ops.SortKey)
def sort_key(op, expr):
    arg = translate(op.expr)
    return arg.sort(ascending=op.ascending)


@translate.register(ops.Selection)
def selection(op, expr):
    plan = translate(op.table)

    selections = []
    for expr in op.selections or [op.table]:
        # TODO(kszucs) it would be nice if we wouldn't need to handle the
        # specific cases in the backend implementations, we could add a
        # new operator which retrieves all of the TableExpr columns
        # (.e.g. Asterisk) so the translate() would handle this
        # automatically
        if isinstance(expr, ir.TableExpr):
            for name in expr.columns:
                column = expr.get_column(name)
                field = translate(column)
                if column.has_name():
                    field = field.alias(column.get_name())
                selections.append(field)
        elif isinstance(expr, ir.ValueExpr):
            field = translate(expr)
            if expr.has_name():
                field = field.alias(expr.get_name())
            selections.append(field)
        else:
            raise com.TranslationError(
                "DataFusion backend is unable to compile selection with "
                f"expression type of {type(expr)}"
            )

    plan = plan.select(*selections)

    if op.predicates:
        predicates = map(translate, op.predicates)
        predicate = functools.reduce(operator.and_, predicates)
        plan = plan.filter(predicate)

    if op.sort_keys:
        sort_keys = map(translate, op.sort_keys)
        plan = plan.sort(*sort_keys)

    return plan


@translate.register(ops.Aggregation)
def aggregation(op, expr):
    table = translate(op.table)
    group_by = [translate(expr) for expr in op.by]

    metrics = []
    for expr in op.metrics:
        agg = translate(expr)
        if expr.has_name():
            agg = agg.alias(expr.get_name())
        metrics.append(agg)

    return table.aggregate(group_by, metrics)


@translate.register(ops.Not)
def invert(op, expr):
    arg = translate(op.arg)
    return ~arg


@translate.register(ops.Abs)
def abs(op, expr):
    arg = translate(op.arg)
    return df.functions.abs(arg)


@translate.register(ops.Ceil)
def ceil(op, expr):
    arg = translate(op.arg)
    return df.functions.ceil(arg).cast(pa.int64())


@translate.register(ops.Floor)
def floor(op, expr):
    arg = translate(op.arg)
    return df.functions.floor(arg).cast(pa.int64())


@translate.register(ops.Round)
def round(op, expr):
    arg = translate(op.arg)
    if op.digits is not None:
        raise com.UnsupportedOperationError(
            'Rounding to specific digits is not supported in datafusion'
        )
    return df.functions.round(arg).cast(pa.int64())


@translate.register(ops.Ln)
def ln(op, expr):
    arg = translate(op.arg)
    return df.functions.ln(arg)


@translate.register(ops.Log2)
def log2(op, expr):
    arg = translate(op.arg)
    return df.functions.log2(arg)


@translate.register(ops.Log10)
def log10(op, expr):
    arg = translate(op.arg)
    return df.functions.log10(arg)


@translate.register(ops.Sqrt)
def sqrt(op, expr):
    arg = translate(op.arg)
    return df.functions.sqrt(arg)


@translate.register(ops.Strip)
def strip(op, expr):
    arg = translate(op.arg)
    return df.functions.trim(arg)


@translate.register(ops.LStrip)
def lstrip(op, expr):
    arg = translate(op.arg)
    return df.functions.ltrim(arg)


@translate.register(ops.RStrip)
def rstrip(op, expr):
    arg = translate(op.arg)
    return df.functions.rtrim(arg)


@translate.register(ops.Lowercase)
def lower(op, expr):
    arg = translate(op.arg)
    return df.functions.lower(arg)


@translate.register(ops.Uppercase)
def upper(op, expr):
    arg = translate(op.arg)
    return df.functions.upper(arg)


@translate.register(ops.Reverse)
def reverse(op, expr):
    arg = translate(op.arg)
    return df.functions.reverse(arg)


@translate.register(ops.StringLength)
def strlen(op, expr):
    arg = translate(op.arg)
    return df.functions.character_length(arg)


@translate.register(ops.Capitalize)
def capitalize(op, expr):
    arg = translate(op.arg)
    return df.functions.initcap(arg)


@translate.register(ops.Substring)
def substring(op, expr):
    arg = translate(op.arg)
    start = translate(op.start + 1)
    length = translate(op.length)
    return df.functions.substr(arg, start, length)


@translate.register(ops.RegexExtract)
def regex_extract(op, expr):
    arg = translate(op.arg)
    pattern = translate(op.pattern)
    return df.functions.regexp_match(arg, pattern)


@translate.register(ops.Repeat)
def repeat(op, expr):
    arg = translate(op.arg)
    times = translate(op.times)
    return df.functions.repeat(arg, times)


@translate.register(ops.LPad)
def lpad(op, expr):
    arg = translate(op.arg)
    length = translate(op.length)
    pad = translate(op.pad)
    return df.functions.lpad(arg, length, pad)


@translate.register(ops.RPad)
def rpad(op, expr):
    arg = translate(op.arg)
    length = translate(op.length)
    pad = translate(op.pad)
    return df.functions.rpad(arg, length, pad)


@translate.register(ops.GreaterEqual)
def ge(op, expr):
    return translate(op.left) >= translate(op.right)


@translate.register(ops.LessEqual)
def le(op, expr):
    return translate(op.left) <= translate(op.right)


@translate.register(ops.Greater)
def gt(op, expr):
    return translate(op.left) > translate(op.right)


@translate.register(ops.Less)
def lt(op, expr):
    return translate(op.left) < translate(op.right)


@translate.register(ops.Equals)
def eq(op, expr):
    return translate(op.left) == translate(op.right)


@translate.register(ops.NotEquals)
def ne(op, expr):
    return translate(op.left) != translate(op.right)


@translate.register(ops.Add)
def add(op, expr):
    return translate(op.left) + translate(op.right)


@translate.register(ops.Subtract)
def sub(op, expr):
    return translate(op.left) - translate(op.right)


@translate.register(ops.Multiply)
def mul(op, expr):
    return translate(op.left) * translate(op.right)


@translate.register(ops.Divide)
def div(op, expr):
    return translate(op.left) / translate(op.right)


@translate.register(ops.FloorDivide)
def floordiv(op, expr):
    return df.functions.floor(translate(op.left) / translate(op.right))


@translate.register(ops.Modulus)
def mod(op, expr):
    return translate(op.left) % translate(op.right)


@translate.register(ops.Sum)
def sum(op, expr):
    arg = translate(op.arg)
    return df.functions.sum(arg)


@translate.register(ops.Min)
def min(op, expr):
    arg = translate(op.arg)
    return df.functions.min(arg)


@translate.register(ops.Max)
def max(op, expr):
    arg = translate(op.arg)
    return df.functions.max(arg)


@translate.register(ops.Mean)
def mean(op, expr):
    arg = translate(op.arg)
    return df.functions.avg(arg)


def _prepare_contains_options(options):
    if isinstance(options, ir.AnyScalar):
        # TODO(kszucs): it would be better if we could pass an arrow
        # ListScalar to datafusions in_list function
        return [df.literal(v) for v in options.op().value]
    else:
        return translate(options)


@translate.register(ops.ValueList)
def value_list(op, expr):
    return list(map(translate, op.values))


@translate.register(ops.Contains)
def contains(op, expr):
    value = translate(op.value)
    options = _prepare_contains_options(op.options)
    return df.functions.in_list(value, options, negated=False)


@translate.register(ops.NotContains)
def not_contains(op, expr):
    value = translate(op.value)
    options = _prepare_contains_options(op.options)
    return df.functions.in_list(value, options, negated=True)


@translate.register(ops.ElementWiseVectorizedUDF)
def elementwise_udf(op, expr):
    udf = df.udf(
        op.func,
        input_types=list(map(to_pyarrow_type, op.input_type)),
        return_type=to_pyarrow_type(op.return_type),
        volatility="volatile",
    )
    args = map(translate, op.func_args)

    return udf(*args)
