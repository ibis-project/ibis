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


def _alias(translator, op):
    # just compile the underlying argument because the naming is handled
    # by the translator for the top level expression
    return translator.translate(op.arg)


def _cast(translator, op):
    arg_ = translator.translate(op.arg)
    type_ = serialize(op.to)

    return f'CAST({arg_!s} AS {type_!s})'


def _between(translator, op):
    arg_, lower_, upper_ = map(translator.translate, op.args)
    return f'{arg_!s} BETWEEN {lower_!s} AND {upper_!s}'


def _negate(translator, op):
    return f"-{_parenthesize(translator, op.arg)}"


def _not(translator, op):
    return f"NOT {_parenthesize(translator, op.arg)}"


def _parenthesize(translator, op):
    # function calls don't need parens
    what_ = translator.translate(op)
    if isinstance(op, (*_binary_infix_ops.keys(), *_unary_ops.keys())):
        return f"({what_})"
    else:
        return what_


def _unary(func_name):
    return _fixed_arity(func_name, 1)


def _extract_epoch_seconds(translator, op):
    return _call(translator, 'toRelativeSecondNum', *op.args)


def _fixed_arity(func_name, arity):
    def formatter(translator, op):
        arg_count = len(op.args)
        if arity != arg_count:
            msg = 'Incorrect number of args {0} instead of {1}'
            raise com.UnsupportedOperationError(msg.format(arg_count, arity))
        return _call(translator, func_name, *op.args)

    return formatter


def _array_index(translator, op):
    arg_ = translator.translate(op.arg)
    index_ = _parenthesize(translator, op.index)

    correct_idx_ = f'if({index_} >= 0, {index_} + 1, {index_})'

    return f'arrayElement({arg_}, {correct_idx_})'


def _array_repeat(translator, op):
    arg_ = _parenthesize(translator, op.arg)
    times_ = _parenthesize(translator, op.times)

    select = 'arrayFlatten(groupArray(arr))'
    from_ = f'(select {arg_} as arr from system.numbers limit {times_})'
    return f'(select {select} from {from_})'


def _array_slice(translator, op):
    start_ = _parenthesize(translator, op.start)
    arg_ = translator.translate(op.arg)

    start_correct_ = f'if({start_} < 0, {start_}, {start_} + 1)'

    if op.stop is not None:
        stop_ = _parenthesize(translator, op.stop)

        cast_arg_ = f'if({arg_} = [], CAST({arg_} AS Array(UInt8)), {arg_})'
        neg_start_ = f'(arrayCount({cast_arg_}) + {start_})'
        diff_fmt = f'greatest(-0, {stop_} - {{}})'.format

        length_ = (
            f'if({stop_} < 0, {stop_}, '
            + f'if({start_} < 0, {diff_fmt(neg_start_)}, {diff_fmt(start_)}))'
        )

        return f'arraySlice({arg_}, {start_correct_}, {length_})'

    return f'arraySlice({arg_}, {start_correct_})'


def _map(translator, op):
    keys_ = translator.translate(op.keys)
    values_ = translator.translate(op.values)
    typ = serialize(op.output_dtype)

    return f"CAST(({keys_}, {values_}), '{typ}')"


def _map_get(translator, op):
    arg_ = translator.translate(op.arg)
    key_ = translator.translate(op.key)
    default_ = translator.translate(op.default)

    return f"if(mapContains({arg_}, {key_}), {arg_}[{key_}], {default_})"


def _agg(func):
    def formatter(translator, op):
        where = getattr(op, "where", None)
        args = tuple(
            arg for arg in op.args if arg is not None and arg is not where
        )
        return _aggregate(translator, func, *args, where=where)

    return formatter


def _count_star(translator, op):
    # zero argument count == count(*), countIf when `where` is not None
    return _aggregate(translator, "count", where=op.where)


def _agg_variance_like(func):
    variants = {'sample': f'{func}Samp', 'pop': f'{func}Pop'}

    def formatter(translator, op):
        *args, how, where = op.args
        return _aggregate(translator, variants[how], *args, where=where)

    return formatter


def _corr(translator, op):
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


def _xor(translator, op):
    left_ = _parenthesize(translator, op.left)
    right_ = _parenthesize(translator, op.right)
    return f'xor({left_}, {right_})'


def _varargs(func_name):
    def varargs_formatter(translator, op):
        return _call(translator, func_name, *op.arg.values)

    return varargs_formatter


def _arbitrary(translator, op):
    functions = {
        None: 'any',
        'first': 'any',
        'last': 'anyLast',
        'heavy': 'anyHeavy',
    }
    return _aggregate(translator, functions[op.how], op.arg, where=op.where)


def _substring(translator, op):
    arg_, start_ = translator.translate(op.arg), translator.translate(op.start)

    # Clickhouse is 1-indexed
    if op.length is None or isinstance(op.length, ops.Literal):
        if op.length is not None:
            length_ = op.length.value
            return f'substring({arg_}, {start_} + 1, {length_})'
        else:
            return f'substring({arg_}, {start_} + 1)'
    else:
        length_ = translator.translate(op.length)
        return f'substring({arg_}, {start_} + 1, {length_})'


def _string_find(translator, op):
    if op.start is not None:
        raise com.UnsupportedOperationError(
            "String find doesn't support start argument"
        )
    if op.end is not None:
        raise com.UnsupportedOperationError(
            "String find doesn't support end argument"
        )

    return _call(translator, 'position', op.arg, op.substr) + ' - 1'


def _regex_extract(translator, op):
    arg_ = translator.translate(op.arg)
    pattern_ = translator.translate(op.pattern)
    index = op.index

    base = f"extractAll(CAST({arg_} AS String), {pattern_})"
    if index is not None:
        return f"{base}[{translator.translate(index)} + 1]"
    return base


def _parse_url(translator, op):
    if op.extract == 'HOST':
        return _call(translator, 'domain', op.arg)
    elif op.extract == 'PROTOCOL':
        return _call(translator, 'protocol', op.arg)
    elif op.extract == 'PATH':
        return _call(translator, 'path', op.arg)
    elif op.extract == 'QUERY':
        if op.key is not None:
            return _call(translator, 'extractURLParameter', op.arg, op.key)
        else:
            return _call(translator, 'queryString', op.arg)
    else:
        raise com.UnsupportedOperationError(
            f'Parse url with extract {op.extract} is not supported'
        )


def _index_of(translator, op):
    needle_ = translator.translate(op.needle)
    values_ = ','.join(map(translator.translate, op.values.values))
    return f"indexOf([{values_}], {needle_}) - 1"


def _sign(translator, op):
    """Workaround for missing sign function"""
    arg_ = translator.translate(op.arg)
    return 'intDivOrZero({0}, abs({0}))'.format(arg_)


def _round(translator, op):
    if op.digits is not None:
        return _call(translator, 'round', op.arg, op.digits)
    else:
        return _call(translator, 'round', op.arg)


def _hash(translator, op):
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

    if op.how not in algorithms:
        raise com.UnsupportedOperationError(
            f'Unsupported hash algorithm {op.how}'
        )

    return _call(translator, op.how, op.arg)


def _log(translator, op):
    if op.base is None:
        func = 'log'
    elif op.base.value == 2:
        func = 'log2'
    elif op.base.value == 10:
        func = 'log10'
    else:
        raise ValueError(f'Base {op.base} for logarithm not supported!')

    return _call(translator, func, op.arg)


def _value_list(translator, op):
    values_ = map(translator.translate, op.values)
    return '({})'.format(', '.join(values_))


def _interval_format(translator, op):
    dtype = op.output_dtype
    if dtype.unit in {'ms', 'us', 'ns'}:
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    return f'INTERVAL {op.value} {dtype.resolution.upper()}'


def _interval_from_integer(translator, op):
    dtype = op.output_dtype
    if dtype.unit in {'ms', 'us', 'ns'}:
        raise com.UnsupportedOperationError(
            "Clickhouse doesn't support subsecond interval resolutions"
        )

    arg_ = translator.translate(op.arg)
    return f'INTERVAL {arg_} {dtype.resolution.upper()}'


def _literal(translator, op):
    value = op.value
    if value is None and op.output_dtype.nullable:
        return _null_literal(translator, op)
    if isinstance(op.output_dtype, dt.Boolean):
        return '1' if value else '0'
    elif isinstance(op.output_dtype, dt.INET):
        v = str(value)
        return f"toIPv6({v!r})" if ':' in v else f"toIPv4({v!r})"
    elif isinstance(op.output_dtype, dt.String):
        return "'{!s}'".format(value.replace("'", "\\'"))
    elif isinstance(op.output_dtype, (dt.Integer, dt.Decimal, dt.Floating)):
        return repr(value)
    elif isinstance(op.output_dtype, dt.Interval):
        return _interval_format(translator, op)
    elif isinstance(op.output_dtype, dt.Timestamp):
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

        if (timezone := op.output_dtype.timezone) is not None:
            args.append(timezone)

        joined_args = ", ".join(map(repr, args))
        return f"{func}({joined_args})"

    elif isinstance(op.output_dtype, dt.Date):
        if isinstance(value, date):
            value = value.strftime('%Y-%m-%d')
        return f"toDate('{value!s}')"
    elif isinstance(op.output_dtype, dt.Array):
        values = ", ".join(_array_literal_values(translator, op))
        return f"[{values}]"
    elif isinstance(op.output_dtype, dt.Map):
        values = ", ".join(_map_literal_values(translator, op))
        return f"map({values})"
    elif isinstance(op.output_dtype, dt.Set):
        return '({})'.format(', '.join(map(repr, value)))
    elif isinstance(op.output_dtype, dt.Struct):
        fields = ", ".join(
            f"{value} as `{key}`" for key, value in op.value.items()
        )
        return f"tuple({fields})"
    else:
        raise NotImplementedError(type(op))


def _array_literal_values(translator, op):
    value_type = op.output_dtype.value_type
    for v in op.value:
        value = ops.Literal(v, dtype=value_type)
        yield _literal(translator, value)


def _map_literal_values(translator, op):
    value_type = op.output_dtype.value_type
    for k, v in op.value.items():
        value = ops.Literal(v, dtype=value_type)
        yield repr(k)
        yield _literal(translator, value)


class _CaseFormatter:
    def __init__(self, translator, base, cases, results, default):
        self.translator = translator
        self.base = base
        self.cases = cases
        self.results = results
        self.default = default

        # HACK
        self.indent = 2
        self.multiline = len(cases.values) > 1
        self.buf = StringIO()

    def _trans(self, expr):
        return self.translator.translate(expr)

    def get_result(self):
        self.buf.seek(0)

        self.buf.write('CASE')
        if self.base is not None:
            base_str = self._trans(self.base)
            self.buf.write(f' {base_str}')

        for case, result in zip(self.cases.values, self.results.values):
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


def _simple_case(translator, op):
    formatter = _CaseFormatter(
        translator, op.base, op.cases, op.results, op.default
    )
    return formatter.get_result()


def _searched_case(translator, op):
    formatter = _CaseFormatter(
        translator, None, op.cases, op.results, op.default
    )
    return formatter.get_result()


def _table_array_view(translator, op):
    ctx = translator.context
    query = ctx.get_compiled_expr(op.table)
    return f'(\n{util.indent(query, ctx.indent)}\n)'


def _timestamp_from_unix(translator, op):
    if op.unit in {'ms', 'us', 'ns'}:
        raise ValueError(f'`{op.unit}` unit is not supported!')

    return _call(translator, 'toDateTime', op.arg)


def _truncate(translator, op):
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
        converter = converters[op.unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            f'Unsupported truncate unit {op.unit}'
        )

    return _call(translator, converter, op.arg)


def _exists_subquery(translator, op):
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


def _table_column(translator, op):
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


def _string_split(translator, op):
    delim = translator.translate(op.delimiter)
    val = translator.translate(op.arg)
    return f"splitByString({delim}, CAST({val} AS String))"


def _string_join(translator, op):
    sep, elements = op.args
    assert isinstance(
        elements, ops.NodeList
    ), f'elements must be a Sequence, got {type(elements)}'
    sep_ = translator.translate(sep)
    elements_ = ', '.join(map(translator.translate, elements.values))
    return f'arrayStringConcat([{elements_}], {sep_})'


def _string_concat(translator, op):
    args_formatted = ", ".join(map(translator.translate, op.arg.values))
    return f"arrayStringConcat([{args_formatted}])"


def _string_like(translator, op):
    return '{} LIKE {}'.format(
        translator.translate(op.arg), translator.translate(op.pattern)
    )


def _string_ilike(translator, op):
    return 'lower({}) LIKE lower({})'.format(
        translator.translate(op.arg),
        translator.translate(op.pattern),
    )


def _group_concat(translator, op):
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


def _string_right(translator, op):
    arg = translator.translate(op.arg)
    nchars = translator.translate(op.nchars)
    return f"substring({arg}, -({nchars}))"


def _cotangent(translator, op):
    arg = translator.translate(op.arg)
    return f"cos({arg}) / sin({arg})"


def _bit_agg(func):
    def compile(translator, op):
        arg_ = translator.translate(op.arg)
        if not isinstance((type := op.arg.output_dtype), dt.UnsignedInteger):
            nbits = type._nbytes * 8
            arg_ = f"reinterpretAsUInt{nbits}({arg_})"

        if (where := op.where) is not None:
            return f"{func}If({arg_}, {translator.translate(where)})"
        else:
            return f"{func}({arg_})"

    return compile


def _array_column(translator, op):
    args = ", ".join(map(translator.translate, op.cols.values))
    return f"[{args}]"


def _struct_column(translator, op):
    args = ", ".join(map(translator.translate, op.values))
    # ClickHouse struct types cannot be nullable
    # (non-nested fields can be nullable)
    struct_type = serialize(op.output_dtype.copy(nullable=False))
    return f"CAST(({args}) AS {struct_type})"


def _clip(translator, op):
    arg = translator.translate(op.arg)

    if (upper := op.upper) is not None:
        arg = f"least({translator.translate(upper)}, {arg})"

    if (lower := op.lower) is not None:
        arg = f"greatest({translator.translate(lower)}, {arg})"

    return arg


def _struct_field(translator, op):
    return f"{translator.translate(op.arg)}.`{op.field}`"


def _nth_value(translator, op):
    arg = translator.translate(op.arg)
    nth = translator.translate(op.nth)
    return f"nth_value({arg}, ({nth}) + 1)"


def _repeat(translator, op):
    arg = translator.translate(op.arg)
    times = translator.translate(op.times)
    return f"repeat({arg}, CAST({times} AS UInt64))"


def _sort_key(translator, op):
    sort_direction = " DESC" * (not op.ascending)
    return f"{translator.translate(op.expr)}{sort_direction}"


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
    ops.SortKey: _sort_key,
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
    ops.CountStar: _count_star,
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
    ops.NodeList: _value_list,
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
    ops.ArrayIndex: _array_index,
    ops.ArrayConcat: _fixed_arity('arrayConcat', 2),
    ops.ArrayRepeat: _array_repeat,
    ops.ArraySlice: _array_slice,
    ops.Map: _map,
    ops.MapGet: _map_get,
    ops.MapContains: _fixed_arity('mapContains', 2),
    ops.MapKeys: _unary('mapKeys'),
    ops.MapValues: _unary('mapValues'),
    ops.MapMerge: _fixed_arity('mapUpdate', 2),
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


def _raise_error(translator, op, *args):
    msg = "Clickhouse backend doesn't support {0} operation!"
    raise com.UnsupportedOperationError(msg.format(type(op)))


def _null_literal(translator, op):
    return 'Null'


def _null_if_zero(translator, op):
    arg_ = translator.translate(op.arg)
    return f'nullIf({arg_}, 0)'


def _zero_if_null(translator, op):
    arg_ = translator.translate(op.arg)
    return f'ifNull({arg_}, 0)'


def _day_of_week_index(translator, op):
    weekdays = 7
    offset = f"toDayOfWeek({translator.translate(op.arg)})"
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
