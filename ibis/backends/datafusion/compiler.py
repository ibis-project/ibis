import functools
import operator

import datafusion as df
import datafusion.functions
import pyarrow as pa

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.datafusion.datatypes import to_pyarrow_type


@functools.singledispatch
def translate(expr):
    raise NotImplementedError(expr)


@translate.register(ir.Expr)
def expression(expr):
    return translate(expr.op(), expr)


@translate.register(ops.Node)
def operation(op, _):
    raise com.OperationNotDefinedError(f'No translation rule for {type(op)}')


@translate.register(ops.DatabaseTable)
def table(op, _):
    name, _, client = op.args
    return client._context.table(name)


@translate.register(ops.Alias)
def alias(op, _):
    arg = translate(op.arg)
    return arg.alias(op.name)


@translate.register(ops.Literal)
def literal(op, _):
    if isinstance(op.value, (set, frozenset)):
        value = list(op.value)
    else:
        value = op.value

    arrow_type = to_pyarrow_type(op.dtype)
    arrow_scalar = pa.scalar(value, type=arrow_type)

    return df.literal(arrow_scalar)


@translate.register(ops.Cast)
def cast(op, _):
    arg = translate(op.arg)
    typ = to_pyarrow_type(op.to)
    return arg.cast(to=typ)


@translate.register(ops.TableColumn)
def column(op, _):
    table_op = op.table.op()

    if hasattr(table_op, "name"):
        return df.column(f'{table_op.name}."{op.name}"')
    else:
        return df.column(op.name)


@translate.register(ops.SortKey)
def sort_key(op, _):
    arg = translate(op.expr)
    return arg.sort(ascending=op.ascending)


@translate.register(ops.Selection)
def selection(op, expr):
    plan = translate(op.table)

    selections = []
    for expr in op.selections or [op.table]:
        # TODO(kszucs) it would be nice if we wouldn't need to handle the
        # specific cases in the backend implementations, we could add a
        # new operator which retrieves all of the Table columns
        # (.e.g. Asterisk) so the translate() would handle this
        # automatically
        if isinstance(expr, ir.Table):
            for name in expr.columns:
                column = expr.get_column(name)
                field = translate(column)
                selections.append(field)
        elif isinstance(expr, ir.Value):
            field = translate(expr)
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
        metrics.append(agg)

    if op.predicates:
        table = table.filter(
            functools.reduce(
                operator.and_,
                map(translate, op.predicates),
            )
        )

    return table.aggregate(group_by, metrics)


@translate.register(ops.Not)
def invert(op, _):
    arg = translate(op.arg)
    return ~arg


@translate.register(ops.Abs)
def abs(op, _):
    arg = translate(op.arg)
    return df.functions.abs(arg)


@translate.register(ops.Ceil)
def ceil(op, _):
    arg = translate(op.arg)
    return df.functions.ceil(arg).cast(pa.int64())


@translate.register(ops.Floor)
def floor(op, _):
    arg = translate(op.arg)
    return df.functions.floor(arg).cast(pa.int64())


@translate.register(ops.Round)
def round(op, _):
    arg = translate(op.arg)
    if op.digits is not None:
        raise com.UnsupportedOperationError(
            'Rounding to specific digits is not supported in datafusion'
        )
    return df.functions.round(arg).cast(pa.int64())


@translate.register(ops.Ln)
def ln(op, _):
    arg = translate(op.arg)
    return df.functions.ln(arg)


@translate.register(ops.Log2)
def log2(op, _):
    arg = translate(op.arg)
    return df.functions.log2(arg)


@translate.register(ops.Log10)
def log10(op, _):
    arg = translate(op.arg)
    return df.functions.log10(arg)


@translate.register(ops.Sqrt)
def sqrt(op, _):
    arg = translate(op.arg)
    return df.functions.sqrt(arg)


@translate.register(ops.Strip)
def strip(op, _):
    arg = translate(op.arg)
    return df.functions.trim(arg)


@translate.register(ops.LStrip)
def lstrip(op, _):
    arg = translate(op.arg)
    return df.functions.ltrim(arg)


@translate.register(ops.RStrip)
def rstrip(op, _):
    arg = translate(op.arg)
    return df.functions.rtrim(arg)


@translate.register(ops.Lowercase)
def lower(op, _):
    arg = translate(op.arg)
    return df.functions.lower(arg)


@translate.register(ops.Uppercase)
def upper(op, _):
    arg = translate(op.arg)
    return df.functions.upper(arg)


@translate.register(ops.Reverse)
def reverse(op, _):
    arg = translate(op.arg)
    return df.functions.reverse(arg)


@translate.register(ops.StringLength)
def strlen(op, _):
    arg = translate(op.arg)
    return df.functions.character_length(arg)


@translate.register(ops.Capitalize)
def capitalize(op, _):
    arg = translate(op.arg)
    return df.functions.initcap(arg)


@translate.register(ops.Substring)
def substring(op, _):
    arg = translate(op.arg)
    start = translate(op.start + 1)
    length = translate(op.length)
    return df.functions.substr(arg, start, length)


@translate.register(ops.RegexExtract)
def regex_extract(op, _):
    arg = translate(op.arg)
    pattern = translate(op.pattern)
    return df.functions.regexp_match(arg, pattern)


@translate.register(ops.Repeat)
def repeat(op, _):
    arg = translate(op.arg)
    times = translate(op.times)
    return df.functions.repeat(arg, times)


@translate.register(ops.LPad)
def lpad(op, _):
    arg = translate(op.arg)
    length = translate(op.length)
    pad = translate(op.pad)
    return df.functions.lpad(arg, length, pad)


@translate.register(ops.RPad)
def rpad(op, _):
    arg = translate(op.arg)
    length = translate(op.length)
    pad = translate(op.pad)
    return df.functions.rpad(arg, length, pad)


@translate.register(ops.GreaterEqual)
def ge(op, _):
    return translate(op.left) >= translate(op.right)


@translate.register(ops.LessEqual)
def le(op, _):
    return translate(op.left) <= translate(op.right)


@translate.register(ops.Greater)
def gt(op, _):
    return translate(op.left) > translate(op.right)


@translate.register(ops.Less)
def lt(op, _):
    return translate(op.left) < translate(op.right)


@translate.register(ops.Equals)
def eq(op, _):
    return translate(op.left) == translate(op.right)


@translate.register(ops.NotEquals)
def ne(op, _):
    return translate(op.left) != translate(op.right)


@translate.register(ops.Add)
def add(op, _):
    return translate(op.left) + translate(op.right)


@translate.register(ops.Subtract)
def sub(op, _):
    return translate(op.left) - translate(op.right)


@translate.register(ops.Multiply)
def mul(op, _):
    return translate(op.left) * translate(op.right)


@translate.register(ops.Divide)
def div(op, _):
    return translate(op.left) / translate(op.right)


@translate.register(ops.FloorDivide)
def floordiv(op, _):
    return df.functions.floor(translate(op.left) / translate(op.right))


@translate.register(ops.Modulus)
def mod(op, _):
    return translate(op.left) % translate(op.right)


@translate.register(ops.Count)
def count(op, _):
    op_arg = op.arg
    if isinstance(op_arg, ir.Table):
        arg = df.literal(1)
    else:
        arg = translate(op_arg)
    return df.functions.count(arg)


@translate.register(ops.Sum)
def sum(op, _):
    arg = translate(op.arg)
    return df.functions.sum(arg)


@translate.register(ops.Min)
def min(op, _):
    arg = translate(op.arg)
    return df.functions.min(arg)


@translate.register(ops.Max)
def max(op, _):
    arg = translate(op.arg)
    return df.functions.max(arg)


@translate.register(ops.Mean)
def mean(op, _):
    arg = translate(op.arg)
    return df.functions.avg(arg)


def _prepare_contains_options(options):
    if isinstance(options, ir.Scalar):
        # TODO(kszucs): it would be better if we could pass an arrow
        # ListScalar to datafusions in_list function
        return [df.literal(v) for v in options.op().value]
    else:
        return translate(options)


@translate.register(ops.ValueList)
def value_list(op, _):
    return list(map(translate, op.values))


@translate.register(ops.Contains)
def contains(op, _):
    value = translate(op.value)
    options = _prepare_contains_options(op.options)
    return df.functions.in_list(value, options, negated=False)


@translate.register(ops.NotContains)
def not_contains(op, _):
    value = translate(op.value)
    options = _prepare_contains_options(op.options)
    return df.functions.in_list(value, options, negated=True)


@translate.register(ops.Negate)
def negate(op, _):
    return df.lit(-1) * translate(op.arg)


@translate.register(ops.Acos)
@translate.register(ops.Asin)
@translate.register(ops.Atan)
@translate.register(ops.Cos)
@translate.register(ops.Sin)
@translate.register(ops.Tan)
def trig(op, _):
    func_name = op.__class__.__name__.lower()
    func = getattr(df.functions, func_name)
    return func(translate(op.arg))


@translate.register(ops.Atan2)
def atan2(op, _):
    y, x = map(translate, op.args)
    return df.functions.atan(y / x)


@translate.register(ops.Cot)
def cot(op, _):
    x = translate(op.arg)
    return df.functions.cos(x) / df.functions.sin(x)


@translate.register(ops.ElementWiseVectorizedUDF)
def elementwise_udf(op, _):
    udf = df.udf(
        op.func,
        input_types=list(map(to_pyarrow_type, op.input_type)),
        return_type=to_pyarrow_type(op.return_type),
        volatility="volatile",
    )
    args = map(translate, op.func_args)

    return udf(*args)
