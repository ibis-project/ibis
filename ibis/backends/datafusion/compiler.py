from __future__ import annotations

import functools
import operator

import datafusion as df
import datafusion.functions
import pyarrow as pa
import pyarrow.compute as pc

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.datafusion.datatypes import to_pyarrow_type


@functools.singledispatch
def translate(expr):
    raise NotImplementedError(expr)


@translate.register(ops.Node)
def operation(op):
    raise com.OperationNotDefinedError(f'No translation rule for {type(op)}')


@translate.register(ops.DatabaseTable)
def table(op):
    name, _, client = op.args
    return client._context.table(name)


@translate.register(ops.Alias)
def alias(op):
    arg = translate(op.arg)
    return arg.alias(op.name)


@translate.register(ops.Literal)
def literal(op):
    if isinstance(op.value, (set, frozenset)):
        value = list(op.value)
    else:
        value = op.value

    arrow_type = to_pyarrow_type(op.dtype)
    arrow_scalar = pa.scalar(value, type=arrow_type)

    return df.literal(arrow_scalar)


@translate.register(ops.Cast)
def cast(op):
    arg = translate(op.arg)
    typ = to_pyarrow_type(op.to)
    return arg.cast(to=typ)


@translate.register(ops.TableColumn)
def column(op):
    table_op = op.table

    if hasattr(table_op, "name"):
        return df.column(f'{table_op.name}."{op.name}"')
    else:
        return df.column(op.name)


@translate.register(ops.SortKey)
def sort_key(op):
    arg = translate(op.expr)
    return arg.sort(ascending=op.ascending)


@translate.register(ops.Selection)
def selection(op):
    plan = translate(op.table)

    selections = []
    for arg in op.selections or [op.table]:
        # TODO(kszucs) it would be nice if we wouldn't need to handle the
        # specific cases in the backend implementations, we could add a
        # new operator which retrieves all of the Table columns
        # (.e.g. Asterisk) so the translate() would handle this
        # automatically
        if isinstance(arg, ops.TableNode):
            for name in arg.schema.names:
                column = ops.TableColumn(table=arg, name=name)
                field = translate(column)
                selections.append(field)
        elif isinstance(arg, ops.Value):
            field = translate(arg)
            selections.append(field)
        else:
            raise com.TranslationError(
                "DataFusion backend is unable to compile selection with "
                f"operation type of {type(arg)}"
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


@translate.register(ops.Limit)
def limit(op):
    if op.offset:
        raise NotImplementedError("DataFusion does not support offset")
    return translate(op.table).limit(op.n)


@translate.register(ops.Aggregation)
def aggregation(op):
    table = translate(op.table)
    group_by = [translate(arg) for arg in op.by]
    metrics = [translate(arg) for arg in op.metrics]

    if op.predicates:
        table = table.filter(
            functools.reduce(
                operator.and_,
                map(translate, op.predicates),
            )
        )

    return table.aggregate(group_by, metrics)


@translate.register(ops.Not)
def invert(op):
    arg = translate(op.arg)
    return ~arg


@translate.register(ops.And)
def and_(op):
    left = translate(op.left)
    right = translate(op.right)
    return left & right


@translate.register(ops.Or)
def or_(op):
    left = translate(op.left)
    right = translate(op.right)
    return left | right


@translate.register(ops.Abs)
def abs(op):
    arg = translate(op.arg)
    return df.functions.abs(arg)


@translate.register(ops.Ceil)
def ceil(op):
    arg = translate(op.arg)
    return df.functions.ceil(arg).cast(pa.int64())


@translate.register(ops.Floor)
def floor(op):
    arg = translate(op.arg)
    return df.functions.floor(arg).cast(pa.int64())


@translate.register(ops.Round)
def round(op):
    arg = translate(op.arg)
    if op.digits is not None:
        raise com.UnsupportedOperationError(
            'Rounding to specific digits is not supported in datafusion'
        )
    return df.functions.round(arg).cast(pa.int64())


@translate.register(ops.Ln)
def ln(op):
    arg = translate(op.arg)
    return df.functions.ln(arg)


@translate.register(ops.Log2)
def log2(op):
    arg = translate(op.arg)
    return df.functions.log2(arg)


@translate.register(ops.Log10)
def log10(op):
    arg = translate(op.arg)
    return df.functions.log10(arg)


@translate.register(ops.Sqrt)
def sqrt(op):
    arg = translate(op.arg)
    return df.functions.sqrt(arg)


@translate.register(ops.Strip)
def strip(op):
    arg = translate(op.arg)
    return df.functions.trim(arg)


@translate.register(ops.LStrip)
def lstrip(op):
    arg = translate(op.arg)
    return df.functions.ltrim(arg)


@translate.register(ops.RStrip)
def rstrip(op):
    arg = translate(op.arg)
    return df.functions.rtrim(arg)


@translate.register(ops.Lowercase)
def lower(op):
    arg = translate(op.arg)
    return df.functions.lower(arg)


@translate.register(ops.Uppercase)
def upper(op):
    arg = translate(op.arg)
    return df.functions.upper(arg)


@translate.register(ops.Reverse)
def reverse(op):
    arg = translate(op.arg)
    return df.functions.reverse(arg)


@translate.register(ops.StringLength)
def strlen(op):
    arg = translate(op.arg)
    return df.functions.character_length(arg)


@translate.register(ops.Capitalize)
def capitalize(op):
    arg = translate(op.arg)
    return df.functions.initcap(arg)


@translate.register(ops.Substring)
def substring(op):
    arg = translate(op.arg)
    start = translate(ops.Add(left=op.start, right=1))
    length = translate(op.length)
    return df.functions.substr(arg, start, length)


@translate.register(ops.Repeat)
def repeat(op):
    arg = translate(op.arg)
    times = translate(op.times)
    return df.functions.repeat(arg, times)


@translate.register(ops.LPad)
def lpad(op):
    arg = translate(op.arg)
    length = translate(op.length)
    pad = translate(op.pad)
    return df.functions.lpad(arg, length, pad)


@translate.register(ops.RPad)
def rpad(op):
    arg = translate(op.arg)
    length = translate(op.length)
    pad = translate(op.pad)
    return df.functions.rpad(arg, length, pad)


@translate.register(ops.GreaterEqual)
def ge(op):
    return translate(op.left) >= translate(op.right)


@translate.register(ops.LessEqual)
def le(op):
    return translate(op.left) <= translate(op.right)


@translate.register(ops.Greater)
def gt(op):
    return translate(op.left) > translate(op.right)


@translate.register(ops.Less)
def lt(op):
    return translate(op.left) < translate(op.right)


@translate.register(ops.Equals)
def eq(op):
    return translate(op.left) == translate(op.right)


@translate.register(ops.NotEquals)
def ne(op):
    return translate(op.left) != translate(op.right)


@translate.register(ops.Add)
def add(op):
    return translate(op.left) + translate(op.right)


@translate.register(ops.Subtract)
def sub(op):
    return translate(op.left) - translate(op.right)


@translate.register(ops.Multiply)
def mul(op):
    return translate(op.left) * translate(op.right)


@translate.register(ops.Divide)
def div(op):
    return translate(op.left) / translate(op.right)


@translate.register(ops.FloorDivide)
def floordiv(op):
    return df.functions.floor(translate(op.left) / translate(op.right))


@translate.register(ops.Modulus)
def mod(op):
    return translate(op.left) % translate(op.right)


@translate.register(ops.Count)
def count(op):
    return df.functions.count(translate(op.arg))


@translate.register(ops.CountStar)
def count_star(_):
    return df.functions.count(df.literal(1))


@translate.register(ops.Sum)
def sum(op):
    arg = translate(op.arg)
    return df.functions.sum(arg)


@translate.register(ops.Min)
def min(op):
    arg = translate(op.arg)
    return df.functions.min(arg)


@translate.register(ops.Max)
def max(op):
    arg = translate(op.arg)
    return df.functions.max(arg)


@translate.register(ops.Mean)
def mean(op):
    arg = translate(op.arg)
    return df.functions.avg(arg)


@translate.register(ops.Contains)
def contains(op):
    value = translate(op.value)
    options = list(map(translate, op.options))
    return df.functions.in_list(value, options, negated=False)


@translate.register(ops.NotContains)
def not_contains(op):
    value = translate(op.value)
    options = list(map(translate, op.options))
    return df.functions.in_list(value, options, negated=True)


@translate.register(ops.Negate)
def negate(op):
    return df.lit(-1) * translate(op.arg)


@translate.register(ops.Acos)
@translate.register(ops.Asin)
@translate.register(ops.Atan)
@translate.register(ops.Cos)
@translate.register(ops.Sin)
@translate.register(ops.Tan)
def trig(op):
    func_name = op.__class__.__name__.lower()
    func = getattr(df.functions, func_name)
    return func(translate(op.arg))


@translate.register(ops.Atan2)
def atan2(op):
    y, x = map(translate, op.args)
    return df.functions.atan(y / x)


@translate.register(ops.Cot)
def cot(op):
    x = translate(op.arg)
    return df.lit(1.0) / df.functions.tan(x)


@translate.register(ops.ElementWiseVectorizedUDF)
def elementwise_udf(op):
    udf = df.udf(
        op.func,
        input_types=list(map(to_pyarrow_type, op.input_type)),
        return_type=to_pyarrow_type(op.return_type),
        volatility="volatile",
    )
    args = map(translate, op.func_args)

    return udf(*args)


@translate.register(ops.StringConcat)
def string_concat(op):
    return df.functions.concat(*map(translate, op.arg))


@translate.register(ops.RegexExtract)
def regex_extract(op):
    arg = translate(op.arg)
    concat = ops.StringConcat(("(", op.pattern, ")"))
    pattern = translate(concat)
    if (index := getattr(op.index, "value", None)) is None:
        raise ValueError(
            "re_extract `index` expressions must be literals. "
            "Arbitrary expressions are not supported in the DataFusion backend"
        )
    string_array_get = df.udf(
        lambda arr, index=index: pc.list_element(arr, index),
        input_types=[to_pyarrow_type(dt.Array(dt.string))],
        return_type=to_pyarrow_type(dt.string),
        volatility="immutable",
        name="string_array_get",
    )
    return string_array_get(df.functions.regexp_match(arg, pattern))
