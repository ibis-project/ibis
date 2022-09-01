from datetime import date, datetime
from io import StringIO

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base.sql.registry import binary_infix, window
from ibis.backends.clickhouse.datatypes import serialize
from ibis.backends.clickhouse.identifiers import quote_identifier

# TODO(kszucs): should inherit operation registry from the base compiler


def _alias(translator, expr):
    # just compile the underlying argument because the naming is handled
    # by the translator for the top level expression
    op = expr.op()
    return translator.translate(op.arg)


def _cast(translator, expr):
    op = expr.op()
    arg = op.arg
    target = op.to
    arg_ = translator.translate(arg)
    type_ = serialize(target)

    return f'CAST({arg_!s} AS {type_!s})'


def _between(translator, expr):
    op = expr.op()
    arg_, lower_, upper_ = map(translator.translate, op.args)
    return f'{arg_!s} BETWEEN {lower_!s} AND {upper_!s}'


def _negate(translator, expr):
    return f"-{_parenthesize(translator, expr.op().arg)}"


def _not(translator, expr):
    return f"NOT {_parenthesize(translator, expr.op().arg)}"


def _parenthesize(translator, expr):
    op = expr.op()

    # function calls don't need parens
    what_ = translator.translate(expr)
    if isinstance(op, (*_binary_infix_ops.keys(), *_unary_ops.keys())):
        return f"({what_})"
    else:
        return what_


def _unary(func_name):
    return _fixed_arity(func_name, 1)


def _extract_epoch_seconds(translator, expr):
    op = expr.op()
    return _call(translator, 'toRelativeSecondNum', *op.args)


def _fixed_arity(func_name, arity):
    def formatter(translator, expr):
        op = expr.op()
        arg_count = len(op.args)
        if arity != arg_count:
            msg = 'Incorrect number of args {0} instead of {1}'
            raise com.UnsupportedOperationError(msg.format(arg_count, arity))
        return _call(translator, func_name, *op.args)

    return formatter


def _array_index_op(translator, expr):
    op = expr.op()

    arr = op.args[0]
    idx = op.args[1]

    arr_ = translator.translate(arr)
    idx_ = _parenthesize(translator, idx)

    correct_idx = f'if({idx_} >= 0, {idx_} + 1, {idx_})'

    return f'arrayElement({arr_}, {correct_idx})'


def _array_repeat_op(translator, expr):
    op = expr.op()
    arr, times = op.args

    arr_ = _parenthesize(translator, arr)
    times_ = _parenthesize(translator, times)

    select = 'arrayFlatten(groupArray(arr))'
    from_ = f'(select {arr_} as arr from system.numbers limit {times_})'
    return f'(select {select} from {from_})'


def _array_slice_op(translator, expr):
    op = expr.op()
    arg, start, stop = op.args

    start_ = _parenthesize(translator, start)
    arg_ = translator.translate(arg)

    start_correct_ = f'if({start_} < 0, {start_}, {start_} + 1)'

    if stop is not None:
        stop_ = _parenthesize(translator, stop)

        cast_arg_ = f'if({arg_} = [], CAST({arg_} AS Array(UInt8)), {arg_})'
        neg_start_ = f'(arrayCount({cast_arg_}) + {start_})'
        diff_fmt = f'greatest(-0, {stop_} - {{}})'.format

        length_ = (
            f'if({stop_} < 0, {stop_}, '
            + f'if({start_} < 0, {diff_fmt(neg_start_)}, {diff_fmt(start_)}))'
        )

        return f'arraySlice({arg_}, {start_correct_}, {length_})'

    return f'arraySlice({arg_}, {start_correct_})'


def _agg(func):
    def formatter(translator, expr):
        op = expr.op()
        where = getattr(op, "where", None)
        args = tuple(
            arg for arg in op.args if arg is not None and arg is not where
        )
        return _aggregate(translator, func, *args, where=where)

    return formatter


def _agg_variance_like(func):
    variants = {'sample': f'{func}Samp', 'pop': f'{func}Pop'}

    def formatter(translator, expr):
        *args, how, where = expr.op().args
        return _aggregate(translator, variants[how], *args, where=where)

    return formatter


def _corr(translator, expr):
    op = expr.op()
    if op.how == "pop":
        raise ValueError(
            "ClickHouse only implements `sample` correlation coefficient"
        )
    return _aggregate(translator, "corr", op.left, op.right, where=op.where)


def _call(translator, func, *args):
    args_ = ', '.join(map(translator.translate, args))
    return f'{func!s}({args_!s})'


def _aggregate(translator, func, *args, where=None):
    if where is not None:
        return _call(translator, f"{func}If", *args, where)
    else:
        return _call(translator, func, *args)


def _xor(translator, expr):
    op = expr.op()
    left_ = _parenthesize(translator, op.left)
    right_ = _parenthesize(translator, op.right)
    return f'xor({left_}, {right_})'


def _varargs(func_name):
    def varargs_formatter(translator, expr):
        op = expr.op()
        return _call(translator, func_name, *op.arg)

    return varargs_formatter


def _arbitrary(translator, expr):
    op = expr.op()
    functions = {
        None: 'any',
        'first': 'any',
        'last': 'anyLast',
        'heavy': 'anyHeavy',
    }
    return _aggregate(translator, functions[op.how], op.arg, where=op.where)


def _substring(translator, expr):
    # arg_ is the formatted notation
    op = expr.op()
    arg, start, length = op.args
    arg_, start_ = translator.translate(arg), translator.translate(start)

    # Clickhouse is 1-indexed
    if length is None or isinstance(length.op(), ops.Literal):
        if length is not None:
            length_ = length.op().value
            return f'substring({arg_}, {start_} + 1, {length_})'
        else:
            return f'substring({arg_}, {start_} + 1)'
    else:
        length_ = translator.translate(length)
        return f'substring({arg_}, {start_} + 1, {length_})'


def _string_find(translator, expr):
    op = expr.op()
    arg, substr, start, _ = op.args
    if start is not None:
        raise com.UnsupportedOperationError(
            "String find doesn't support start argument"
        )

    return _call(translator, 'position', arg, substr) + ' - 1'


def _regex_extract(translator, expr):
    op = expr.op()
    arg_ = translator.translate(op.arg)
    pattern_ = translator.translate(op.pattern)
    index = op.index

    base = f"extractAll(CAST({arg_} AS String), {pattern_})"
    if index is not None:
        return f"{base}[{translator.translate(index)} + 1]"
    return base


def _parse_url(translator, expr):
    op = expr.op()
    arg, extract, key = op.args

    if extract == 'HOST':
        return _call(translator, 'domain', arg)
    elif extract == 'PROTOCOL':
        return _call(translator, 'protocol', arg)
    elif extract == 'PATH':
        return _call(translator, 'path', arg)
    elif extract == 'QUERY':
        if key is not None:
            return _call(translator, 'extractURLParameter', arg, key)
        else:
            return _call(translator, 'queryString', arg)
    else:
        raise com.UnsupportedOperationError(
            f'Parse url with extract {extract} is not supported'
        )


def _index_of(translator, expr):
    op = expr.op()

    arg, arr = op.args
    arg_formatted = translator.translate(arg)
    arr_formatted = ','.join(map(translator.translate, arr))
    return f"indexOf([{arr_formatted}], {arg_formatted}) - 1"


def _sign(translator, expr):
    """Workaround for missing sign function"""
    op = expr.op()
    (arg,) = op.args
    arg_ = translator.translate(arg)
    return 'intDivOrZero({0}, abs({0}))'.format(arg_)


def _round(translator, expr):
    op = expr.op()
    arg, digits = op.args

    if digits is not None:
        return _call(translator, 'round', arg, digits)
    else:
        return _call(translator, 'round', arg)


def _hash(translator, expr):
    op = expr.op()
    arg, how = op.args

    algorithms = {
        'MD5',
        'halfMD5',
        'SHA1',
        'SHA224',
        'SHA256',
        'intHash32',
        'intHash64',
        'cityHash64',
        'sipHash64',
        'sipHash128',
    }

    if how not in algorithms:
        raise com.UnsupportedOperationError(
            f'Unsupported hash algorithm {how}'
        )

    return _call(translator, how, arg)


def _log(translator, expr):
    op = expr.op()
    arg, base = op.args

    if base is None:
        func = 'log'
    elif base._arg.value == 2:
        func = 'log2'
    elif base._arg.value == 10:
        func = 'log10'
    else:
        raise ValueError(f'Base {base} for logarithm not supported!')

    return _call(translator, func, arg)


def _value_list(translator, expr):
    op = expr.op()
    values_ = map(translator.translate, op.values)
    return '({})'.format(', '.join(values_))


def _interval_format(translator, expr):
    dtype = expr.type()
    if dtype.unit in {'ms', 'us', 'ns'}:
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    return f'INTERVAL {expr.op().value} {dtype.resolution.upper()}'


def _interval_from_integer(translator, expr):
    op = expr.op()
    arg, unit = op.args

    dtype = expr.type()
    if dtype.unit in {'ms', 'us', 'ns'}:
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    arg_ = translator.translate(arg)
    return f'INTERVAL {arg_} {dtype.resolution.upper()}'


def _literal(translator, expr):
    value = expr.op().value
    if value is None and expr.type().nullable:
        return _null_literal(translator, expr)
    if isinstance(expr, ir.BooleanValue):
        return '1' if value else '0'
    elif isinstance(expr, ir.INETValue):
        v = str(value)
        return f"toIPv6({v!r})" if ':' in v else f"toIPv4({v!r})"
    elif isinstance(expr, ir.StringValue):
        return "'{!s}'".format(value.replace("'", "\\'"))
    elif isinstance(expr, ir.NumericValue):
        return repr(value)
    elif isinstance(expr, ir.IntervalValue):
        return _interval_format(translator, expr)
    elif isinstance(expr, ir.TimestampValue):
        func = "toDateTime"
        args = []

        if isinstance(value, datetime):
            fmt = "%Y-%m-%dT%H:%M:%S"

            if micros := value.microsecond:
                func = "toDateTime64"
                fmt += ".%f"

            args.append(value.strftime(fmt))
            if micros % 1000:
                args.append(6)
            elif micros // 1000:
                args.append(3)
        else:
            args.append(str(value))

        if (timezone := expr.type().timezone) is not None:
            args.append(timezone)

        joined_args = ", ".join(map(repr, args))
        return f"{func}({joined_args})"

    elif isinstance(expr, ir.DateValue):
        if isinstance(value, date):
            value = value.strftime('%Y-%m-%d')
        return f"toDate('{value!s}')"
    elif isinstance(expr, ir.ArrayValue):
        return str(list(_tuple_to_list(value)))
    elif isinstance(expr, ir.SetScalar):
        return '({})'.format(', '.join(map(repr, value)))
    elif isinstance(expr, ir.StructScalar):
        fields = ", ".join(
            f"{value} as `{key}`" for key, value in expr.op().value.items()
        )
        return f"tuple({fields})"
    else:
        raise NotImplementedError(type(expr))


def _tuple_to_list(t: tuple):
    for element in t:
        if util.is_iterable(element):
            yield list(_tuple_to_list(element))
        else:
            yield element


class _CaseFormatter:
    def __init__(self, translator, base, cases, results, default):
        self.translator = translator
        self.base = base
        self.cases = cases
        self.results = results
        self.default = default

        # HACK
        self.indent = 2
        self.multiline = len(cases) > 1
        self.buf = StringIO()

    def _trans(self, expr):
        return self.translator.translate(expr)

    def get_result(self):
        self.buf.seek(0)

        self.buf.write('CASE')
        if self.base is not None:
            base_str = self._trans(self.base)
            self.buf.write(f' {base_str}')

        for case, result in zip(self.cases, self.results):
            self._next_case()
            case_str = self._trans(case)
            result_str = self._trans(result)
            self.buf.write(f'WHEN {case_str} THEN {result_str}')

        if self.default is not None:
            self._next_case()
            default_str = self._trans(self.default)
            self.buf.write(f'ELSE {default_str}')

        if self.multiline:
            self.buf.write('\nEND')
        else:
            self.buf.write(' END')

        return self.buf.getvalue()

    def _next_case(self):
        if self.multiline:
            self.buf.write('\n{}'.format(' ' * self.indent))
        else:
            self.buf.write(' ')


def _simple_case(translator, expr):
    op = expr.op()
    formatter = _CaseFormatter(
        translator, op.base, op.cases, op.results, op.default
    )
    return formatter.get_result()


def _searched_case(translator, expr):
    op = expr.op()
    formatter = _CaseFormatter(
        translator, None, op.cases, op.results, op.default
    )
    return formatter.get_result()


def _table_array_view(translator, expr):
    ctx = translator.context
    table = expr.op().table
    query = ctx.get_compiled_expr(table)
    return f'(\n{util.indent(query, ctx.indent)}\n)'


def _timestamp_from_unix(translator, expr):
    op = expr.op()
    arg, unit = op.args

    if unit in {'ms', 'us', 'ns'}:
        raise ValueError(f'`{unit}` unit is not supported!')

    return _call(translator, 'toDateTime', arg)


def _truncate(translator, expr):
    op = expr.op()
    arg, unit = op.args

    converters = {
        'Y': 'toStartOfYear',
        'M': 'toStartOfMonth',
        'W': 'toMonday',
        'D': 'toDate',
        'h': 'toStartOfHour',
        'm': 'toStartOfMinute',
        's': 'toDateTime',
    }

    try:
        converter = converters[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            f'Unsupported truncate unit {unit}'
        )

    return _call(translator, converter, arg)


def _exists_subquery(translator, expr):
    op = expr.op()
    ctx = translator.context

    dummy = ir.literal(1).name(ir.core.unnamed)

    filtered = op.foreign_table.filter(op.predicates)
    expr = filtered.projection([dummy])

    subquery = ctx.get_compiled_expr(expr)

    if isinstance(op, ops.ExistsSubquery):
        key = 'EXISTS'
    elif isinstance(op, ops.NotExistsSubquery):
        key = 'NOT EXISTS'
    else:
        raise NotImplementedError

    return f'{key} (\n{util.indent(subquery, ctx.indent)}\n)'


def _table_column(translator, expr):
    op = expr.op()
    field_name = op.name
    quoted_name = quote_identifier(field_name, force=True)
    table = op.table
    ctx = translator.context

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if translator.permit_subquery and ctx.is_foreign_expr(table):
        proj_expr = table.projection([field_name]).to_array()
        return _table_array_view(translator, proj_expr)

    if ctx.need_aliases():
        alias = ctx.get_ref(table)
        if alias is not None:
            quoted_name = f'{alias}.{quoted_name}'

    return quoted_name


def _string_split(translator, expr):
    op = expr.op()
    delim = translator.translate(op.delimiter)
    val = translator.translate(op.arg)
    return f"splitByString({delim}, CAST({val} AS String))"


def _string_join(translator, expr):
    sep, elements = expr.op().args
    assert isinstance(
        elements.op(), ops.ValueList
    ), f'elements must be a ValueList, got {type(elements.op())}'
    return 'arrayStringConcat([{}], {})'.format(
        ', '.join(map(translator.translate, elements)),
        translator.translate(sep),
    )


def _string_concat(translator, expr):
    args = expr.op().arg
    args_formatted = ", ".join(map(translator.translate, args))
    return f"arrayStringConcat([{args_formatted}])"


def _string_like(translator, expr):
    value, pattern = expr.op().args[:2]
    return '{} LIKE {}'.format(
        translator.translate(value), translator.translate(pattern)
    )


def _string_ilike(translator, expr):
    op = expr.op()
    return 'lower({}) LIKE lower({})'.format(
        translator.translate(op.arg),
        translator.translate(op.pattern),
    )


def _group_concat(translator, expr):
    op = expr.op()

    arg = translator.translate(op.arg)
    sep = translator.translate(op.sep)

    translated_args = [arg]
    func = "groupArray"

    if (where := op.where) is not None:
        func += "If"
        translated_args.append(translator.translate(where))

    call = f"{func}({', '.join(translated_args)})"
    expr = f"arrayStringConcat({call}, {sep})"
    return f"CASE WHEN empty({call}) THEN NULL ELSE {expr} END"


def _string_right(translator, expr):
    op = expr.op()
    arg = translator.translate(op.arg)
    nchars = translator.translate(op.nchars)
    return f"substring({arg}, -({nchars}))"


def _cotangent(translator, expr):
    op = expr.op()
    arg = translator.translate(op.arg)
    return f"cos({arg}) / sin({arg})"


def _bit_agg(func):
    def compile(translator, expr):
        op = expr.op()
        raw_arg = op.arg
        arg = translator.translate(raw_arg)
        if not isinstance((type := raw_arg.type()), dt.UnsignedInteger):
            nbits = type._nbytes * 8
            arg = f"reinterpretAsUInt{nbits}({arg})"

        if (where := op.where) is not None:
            return f"{func}If({arg}, {translator.translate(where)})"
        else:
            return f"{func}({arg})"

    return compile


def _array_column(translator, expr):
    args = ", ".join(map(translator.translate, expr.op().cols))
    return f"[{args}]"


def _struct_column(translator, expr):
    args = ", ".join(map(translator.translate, expr.op().values))
    # ClickHouse struct types cannot be nullable
    # (non-nested fields can be nullable)
    struct_type = serialize(expr.type()(nullable=False))
    return f"CAST(({args}) AS {struct_type})"


def _clip(translator, expr):
    op = expr.op()
    arg = translator.translate(op.arg)

    if (upper := op.upper) is not None:
        arg = f"least({translator.translate(upper)}, {arg})"

    if (lower := op.lower) is not None:
        arg = f"greatest({translator.translate(lower)}, {arg})"

    return arg


def _struct_field(translator, expr):
    op = expr.op()
    return f"{translator.translate(op.arg)}.`{op.field}`"


def _nth_value(translator, expr):
    op = expr.op()
    arg = translator.translate(op.arg)
    nth = translator.translate(op.nth)
    return f"nth_value({arg}, ({nth}) + 1)"


def _repeat(translator, expr):
    op = expr.op()
    arg = translator.translate(op.arg)
    times = translator.translate(op.times)
    return f"repeat({arg}, CAST({times} AS UInt64))"


# TODO: clickhouse uses different string functions
#       for ascii and utf-8 encodings,

_binary_infix_ops = {
    # Binary operations
    ops.Add: binary_infix.binary_infix_op('+'),
    ops.Subtract: binary_infix.binary_infix_op('-'),
    ops.Multiply: binary_infix.binary_infix_op('*'),
    ops.Divide: binary_infix.binary_infix_op('/'),
    ops.Power: _fixed_arity('pow', 2),
    ops.Modulus: binary_infix.binary_infix_op('%'),
    # Comparisons
    ops.Equals: binary_infix.binary_infix_op('='),
    ops.NotEquals: binary_infix.binary_infix_op('!='),
    ops.GreaterEqual: binary_infix.binary_infix_op('>='),
    ops.Greater: binary_infix.binary_infix_op('>'),
    ops.LessEqual: binary_infix.binary_infix_op('<='),
    ops.Less: binary_infix.binary_infix_op('<'),
    # Boolean comparisons
    ops.And: binary_infix.binary_infix_op('AND'),
    ops.Or: binary_infix.binary_infix_op('OR'),
    ops.Xor: _xor,
}

_unary_ops = {ops.Negate: _negate, ops.Not: _not}


operation_registry = {
    ops.Alias: _alias,
    # Unary operations
    ops.TypeOf: _unary('toTypeName'),
    ops.IsNan: _unary('isNaN'),
    ops.IsInf: _unary('isInfinite'),
    ops.Abs: _unary('abs'),
    ops.Ceil: _unary('ceil'),
    ops.Floor: _unary('floor'),
    ops.Exp: _unary('exp'),
    ops.Round: _round,
    ops.Sign: _sign,
    ops.Sqrt: _unary('sqrt'),
    ops.Hash: _hash,
    ops.Log: _log,
    ops.Ln: _unary('log'),
    ops.Log2: _unary('log2'),
    ops.Log10: _unary('log10'),
    ops.Acos: _unary("acos"),
    ops.Asin: _unary("asin"),
    ops.Atan: _unary("atan"),
    ops.Atan2: _fixed_arity("atan2", 2),
    ops.Cos: _unary("cos"),
    ops.Cot: _cotangent,
    ops.Sin: _unary("sin"),
    ops.Tan: _unary("tan"),
    ops.Pi: _fixed_arity("pi", 0),
    ops.E: _fixed_arity("e", 0),
    # Unary aggregates
    ops.CMSMedian: _agg('median'),
    ops.ApproxMedian: _agg('median'),
    # TODO: there is also a `uniq` function which is the
    #       recommended way to approximate cardinality
    ops.HLLCardinality: _agg('uniqHLL12'),
    ops.ApproxCountDistinct: _agg('uniqHLL12'),
    ops.Mean: _agg('avg'),
    ops.Sum: _agg('sum'),
    ops.Max: _agg('max'),
    ops.Min: _agg('min'),
    ops.ArgMin: _agg('argMin'),
    ops.ArgMax: _agg('argMax'),
    ops.ArrayCollect: _agg('groupArray'),
    ops.StandardDev: _agg_variance_like('stddev'),
    ops.Variance: _agg_variance_like('var'),
    ops.Covariance: _agg_variance_like('covar'),
    ops.Correlation: _corr,
    ops.GroupConcat: _group_concat,
    ops.Count: _agg('count'),
    ops.CountDistinct: _agg('uniq'),
    ops.Arbitrary: _arbitrary,
    # string operations
    ops.StringLength: _unary('length'),
    ops.Lowercase: _unary('lower'),
    ops.Uppercase: _unary('upper'),
    ops.Reverse: _unary('reverse'),
    ops.Substring: _substring,
    ops.StringFind: _string_find,
    ops.FindInSet: _index_of,
    ops.StringReplace: _fixed_arity('replaceAll', 3),
    ops.StringJoin: _string_join,
    ops.StringSplit: _string_split,
    ops.StringSQLLike: _string_like,
    ops.StringSQLILike: _string_ilike,
    ops.StartsWith: _fixed_arity("startsWith", 2),
    ops.EndsWith: _fixed_arity("endsWith", 2),
    ops.LPad: _fixed_arity("leftPad", 3),
    ops.RPad: _fixed_arity("rightPad", 3),
    ops.LStrip: _unary('trimLeft'),
    ops.RStrip: _unary('trimRight'),
    ops.Strip: _unary('trimBoth'),
    ops.Repeat: _repeat,
    ops.StringConcat: _string_concat,
    ops.RegexSearch: _fixed_arity('match', 2),
    ops.RegexExtract: _regex_extract,
    ops.RegexReplace: _fixed_arity('replaceRegexpAll', 3),
    ops.ParseURL: _parse_url,
    ops.StrRight: _string_right,
    # Temporal operations
    ops.Date: _unary('toDate'),
    ops.DateTruncate: _truncate,
    ops.TimestampNow: lambda *_: 'now()',
    ops.TimestampTruncate: _truncate,
    ops.TimeTruncate: _truncate,
    ops.IntervalFromInteger: _interval_from_integer,
    ops.ExtractYear: _unary('toYear'),
    ops.ExtractMonth: _unary('toMonth'),
    ops.ExtractDay: _unary('toDayOfMonth'),
    ops.ExtractDayOfYear: _unary('toDayOfYear'),
    ops.ExtractQuarter: _unary('toQuarter'),
    ops.ExtractEpochSeconds: _extract_epoch_seconds,
    ops.ExtractWeekOfYear: _unary('toISOWeek'),
    ops.ExtractHour: _unary('toHour'),
    ops.ExtractMinute: _unary('toMinute'),
    ops.ExtractSecond: _unary('toSecond'),
    # Other operations
    ops.E: lambda *_: 'e()',
    ops.Literal: _literal,
    ops.ValueList: _value_list,
    ops.Cast: _cast,
    # for more than 2 args this should be arrayGreatest|Least(array([]))
    # because clickhouse's greatest and least doesn't support varargs
    ops.Greatest: _varargs('greatest'),
    ops.Least: _varargs('least'),
    ops.Where: _fixed_arity('if', 3),
    ops.Between: _between,
    ops.SimpleCase: _simple_case,
    ops.SearchedCase: _searched_case,
    ops.TableColumn: _table_column,
    ops.TableArrayView: _table_array_view,
    ops.DateAdd: binary_infix.binary_infix_op('+'),
    ops.DateSub: binary_infix.binary_infix_op('-'),
    ops.DateDiff: binary_infix.binary_infix_op('-'),
    ops.Contains: binary_infix.contains("IN"),
    ops.NotContains: binary_infix.contains("NOT IN"),
    ops.TimestampAdd: binary_infix.binary_infix_op('+'),
    ops.TimestampSub: binary_infix.binary_infix_op('-'),
    ops.TimestampDiff: binary_infix.binary_infix_op('-'),
    ops.TimestampFromUNIX: _timestamp_from_unix,
    ops.ExistsSubquery: _exists_subquery,
    ops.NotExistsSubquery: _exists_subquery,
    ops.ArrayLength: _unary('length'),
    ops.ArrayIndex: _array_index_op,
    ops.ArrayConcat: _fixed_arity('arrayConcat', 2),
    ops.ArrayRepeat: _array_repeat_op,
    ops.ArraySlice: _array_slice_op,
    ops.Unnest: _unary("arrayJoin"),
    ops.BitAnd: _bit_agg("groupBitAnd"),
    ops.BitOr: _bit_agg("groupBitOr"),
    ops.BitXor: _bit_agg("groupBitXor"),
    ops.Degrees: _unary("degrees"),
    ops.Radians: _unary("radians"),
    ops.Strftime: _fixed_arity("formatDateTime", 2),
    ops.ArrayColumn: _array_column,
    ops.Clip: _clip,
    ops.StructField: _struct_field,
    ops.StructColumn: _struct_column,
    ops.Window: window.window,
    ops.RowNumber: lambda *args: 'row_number()',
    ops.DenseRank: lambda *args: 'dense_rank()',
    ops.MinRank: lambda *args: 'rank()',
    ops.Lag: window.shift_like('lagInFrame'),
    ops.Lead: window.shift_like('leadInFrame'),
    ops.FirstValue: _unary('first_value'),
    ops.LastValue: _unary('last_value'),
    ops.NthValue: _nth_value,
    ops.Window: window.window,
    ops.NTile: window.ntile,
    ops.BitwiseAnd: _fixed_arity('bitAnd', 2),
    ops.BitwiseOr: _fixed_arity('bitOr', 2),
    ops.BitwiseXor: _fixed_arity('bitXor', 2),
    ops.BitwiseNot: _unary('bitNot'),
    ops.BitwiseLeftShift: _fixed_arity('bitShiftLeft', 2),
    ops.BitwiseRightShift: _fixed_arity('bitShiftRight', 2),
}


def _raise_error(translator, expr, *args):
    msg = "Clickhouse backend doesn't support {0} operation!"
    op = expr.op()
    raise com.UnsupportedOperationError(msg.format(type(op)))


def _null_literal(translator, expr):
    return 'Null'


def _null_if_zero(translator, expr):
    op = expr.op()
    arg = op.args[0]
    arg_ = translator.translate(arg)
    return f'nullIf({arg_}, 0)'


def _zero_if_null(translator, expr):
    op = expr.op()
    arg = op.args[0]
    arg_ = translator.translate(arg)
    return f'ifNull({arg_}, 0)'


def _day_of_week_index(translator, expr):
    (arg,) = expr.op().args
    weekdays = 7
    offset = f"toDayOfWeek({translator.translate(arg)})"
    return f"((({offset} - 1) % {weekdays:d}) + {weekdays:d}) % {weekdays:d}"


_undocumented_operations = {
    ops.NullLiteral: _null_literal,  # undocumented
    ops.IsNull: _unary('isNull'),
    ops.NotNull: _unary('isNotNull'),
    ops.IfNull: _fixed_arity('ifNull', 2),
    ops.NullIf: _fixed_arity('nullIf', 2),
    ops.Coalesce: _varargs('coalesce'),
    ops.NullIfZero: _null_if_zero,
    ops.ZeroIfNull: _zero_if_null,
    ops.DayOfWeekIndex: _day_of_week_index,
}


_unsupported_ops_list = [
    ops.DecimalPrecision,
    ops.DecimalScale,
    ops.BaseConvert,
    ops.IdenticalTo,
    ops.CumeDist,
    ops.PercentRank,
    ops.ReductionVectorizedUDF,
]
_unsupported_ops = {k: _raise_error for k in _unsupported_ops_list}


operation_registry.update(_undocumented_operations)
operation_registry.update(_unsupported_ops)
operation_registry.update(_unary_ops)
operation_registry.update(_binary_infix_ops)
