from datetime import date, datetime
from io import StringIO

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util

from .identifiers import quote_identifier


def _cast(translator, expr):
    from .client import ClickhouseDataType

    op = expr.op()
    arg, target = op.args
    arg_ = translator.translate(arg)
    type_ = str(ClickhouseDataType.from_ibis(target, nullable=False))

    return f'CAST({arg_!s} AS {type_!s})'


def _between(translator, expr):
    op = expr.op()
    arg_, lower_, upper_ = map(translator.translate, op.args)
    return f'{arg_!s} BETWEEN {lower_!s} AND {upper_!s}'


def _negate(translator, expr):
    arg = expr.op().args[0]
    if isinstance(expr, ir.BooleanValue):
        arg_ = translator.translate(arg)
        return f'NOT {arg_!s}'
    else:
        arg_ = _parenthesize(translator, arg)
        return f'-{arg_!s}'


def _not(translator, expr):
    return 'NOT {}'.format(*map(translator.translate, expr.op().args))


def _parenthesize(translator, expr):
    op = expr.op()
    op_klass = type(op)

    # function calls don't need parens
    what_ = translator.translate(expr)
    if (op_klass in _binary_infix_ops) or (op_klass in _unary_ops):
        return f'({what_!s})'
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


def _agg(func):
    def formatter(translator, expr):
        return _aggregate(translator, func, *expr.op().args)

    return formatter


def _agg_variance_like(func):
    variants = {'sample': f'{func}Samp', 'pop': f'{func}Pop'}

    def formatter(translator, expr):
        arg, how, where = expr.op().args
        return _aggregate(translator, variants[how], arg, where)

    return formatter


def _binary_infix_op(infix_sym):
    def formatter(translator, expr):
        op = expr.op()

        left, right = op.args
        left_ = _parenthesize(translator, left)
        right_ = _parenthesize(translator, right)

        return f'{left_!s} {infix_sym!s} {right_!s}'

    return formatter


def _call(translator, func, *args):
    args_ = ', '.join(map(translator.translate, args))
    return f'{func!s}({args_!s})'


def _aggregate(translator, func, arg, where=None):
    if where is not None:
        return _call(translator, func + 'If', arg, where)
    else:
        return _call(translator, func, arg)


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
    arg, how, where = expr.op().args
    functions = {
        None: 'any',
        'first': 'any',
        'last': 'anyLast',
        'heavy': 'anyHeavy',
    }
    return _aggregate(translator, functions[how], arg, where=where)


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
    arg, pattern, index = op.args
    arg_, pattern_ = translator.translate(arg), translator.translate(pattern)

    if index is not None:
        index_ = translator.translate(index)
        return f'extractAll({arg_}, {pattern_})[{index_} + 1]'

    return f'extractAll({arg_}, {pattern_})'


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
    elif isinstance(expr, ir.StringValue):
        return "'{!s}'".format(value.replace("'", "\\'"))
    elif isinstance(expr, ir.NumericValue):
        return repr(value)
    elif isinstance(expr, ir.IntervalValue):
        return _interval_format(translator, expr)
    elif isinstance(expr, ir.TimestampValue):
        if isinstance(value, datetime):
            if value.microsecond != 0:
                msg = 'Unsupported subsecond accuracy {}'
                raise ValueError(msg.format(value))
            value = value.strftime('%Y-%m-%d %H:%M:%S')
        return f"toDateTime('{value!s}')"
    elif isinstance(expr, ir.DateValue):
        if isinstance(value, date):
            value = value.strftime('%Y-%m-%d')
        return f"toDate('{value!s}')"
    elif isinstance(expr, ir.ArrayValue):
        return str(list(value))
    elif isinstance(expr, ir.SetScalar):
        return '({})'.format(', '.join(map(repr, value)))
    else:
        raise NotImplementedError(type(expr))


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

    dummy = ir.literal(1).name(ir.unnamed)

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
    value, sep = expr.op().args
    return 'splitByString({}, {})'.format(
        translator.translate(sep), translator.translate(value)
    )


def _string_join(translator, expr):
    sep, elements = expr.op().args
    assert isinstance(
        elements.op(), ops.ValueList
    ), f'elements must be a ValueList, got {type(elements.op())}'
    return 'arrayStringConcat([{}], {})'.format(
        ', '.join(map(translator.translate, elements)),
        translator.translate(sep),
    )


def _string_repeat(translator, expr):
    value, times = expr.op().args
    result = 'arrayStringConcat(arrayMap(x -> {}, range({})))'.format(
        translator.translate(value), translator.translate(times)
    )
    return result


def _string_like(translator, expr):
    value, pattern = expr.op().args[:2]
    return '{} LIKE {}'.format(
        translator.translate(value), translator.translate(pattern)
    )


def _group_concat(translator, expr):
    arg, sep, where = expr.op().args
    if where is not None:
        arg = where.ifelse(arg, ibis.NA)
    return 'arrayStringConcat(groupArray({}), {})'.format(
        *map(translator.translate, (arg, sep))
    )


# TODO: clickhouse uses different string functions
#       for ascii and utf-8 encodings,

_binary_infix_ops = {
    # Binary operations
    ops.Add: _binary_infix_op('+'),
    ops.Subtract: _binary_infix_op('-'),
    ops.Multiply: _binary_infix_op('*'),
    ops.Divide: _binary_infix_op('/'),
    ops.Power: _fixed_arity('pow', 2),
    ops.Modulus: _binary_infix_op('%'),
    # Comparisons
    ops.Equals: _binary_infix_op('='),
    ops.NotEquals: _binary_infix_op('!='),
    ops.GreaterEqual: _binary_infix_op('>='),
    ops.Greater: _binary_infix_op('>'),
    ops.LessEqual: _binary_infix_op('<='),
    ops.Less: _binary_infix_op('<'),
    # Boolean comparisons
    ops.And: _binary_infix_op('AND'),
    ops.Or: _binary_infix_op('OR'),
    ops.Xor: _xor,
}

_unary_ops = {ops.Negate: _negate, ops.Not: _not}


operation_registry = {
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
    # Unary aggregates
    ops.CMSMedian: _agg('median'),
    # TODO: there is also a `uniq` function which is the
    #       recommended way to approximate cardinality
    ops.HLLCardinality: _agg('uniqHLL12'),
    ops.Mean: _agg('avg'),
    ops.Sum: _agg('sum'),
    ops.Max: _agg('max'),
    ops.Min: _agg('min'),
    ops.StandardDev: _agg_variance_like('stddev'),
    ops.Variance: _agg_variance_like('var'),
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
    ops.Repeat: _string_repeat,
    ops.RegexSearch: _fixed_arity('match', 2),
    # TODO: extractAll(haystack, pattern)[index + 1]
    ops.RegexExtract: _regex_extract,
    ops.RegexReplace: _fixed_arity('replaceRegexpAll', 3),
    ops.ParseURL: _parse_url,
    # Temporal operations
    ops.Date: _unary('toDate'),
    ops.DateTruncate: _truncate,
    ops.TimestampNow: lambda *args: 'now()',
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
    ops.E: lambda *args: 'e()',
    ops.Literal: _literal,
    ops.ValueList: _value_list,
    ops.Cast: _cast,
    # for more than 2 args this should be arrayGreatest|Least(array([]))
    # because clickhouse's greatest and least doesn't support varargs
    ops.Greatest: _varargs('greatest'),
    ops.Least: _varargs('least'),
    ops.Where: _fixed_arity('if', 3),
    ops.Between: _between,
    ops.Contains: _binary_infix_op('IN'),
    ops.NotContains: _binary_infix_op('NOT IN'),
    ops.SimpleCase: _simple_case,
    ops.SearchedCase: _searched_case,
    ops.TableColumn: _table_column,
    ops.TableArrayView: _table_array_view,
    ops.DateAdd: _binary_infix_op('+'),
    ops.DateSub: _binary_infix_op('-'),
    ops.DateDiff: _binary_infix_op('-'),
    ops.TimestampAdd: _binary_infix_op('+'),
    ops.TimestampSub: _binary_infix_op('-'),
    ops.TimestampDiff: _binary_infix_op('-'),
    ops.TimestampFromUNIX: _timestamp_from_unix,
    ops.ExistsSubquery: _exists_subquery,
    ops.NotExistsSubquery: _exists_subquery,
    ops.ArrayLength: _unary('length'),
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
    ops.WindowOp,
    ops.DecimalPrecision,
    ops.DecimalScale,
    ops.BaseConvert,
    ops.CumulativeSum,
    ops.CumulativeMin,
    ops.CumulativeMax,
    ops.CumulativeMean,
    ops.CumulativeAny,
    ops.CumulativeAll,
    ops.IdenticalTo,
    ops.RowNumber,
    ops.DenseRank,
    ops.MinRank,
    ops.PercentRank,
    ops.FirstValue,
    ops.LastValue,
    ops.NthValue,
    ops.Lag,
    ops.Lead,
    ops.NTile,
]
_unsupported_ops = {k: _raise_error for k in _unsupported_ops_list}


operation_registry.update(_undocumented_operations)
operation_registry.update(_unsupported_ops)
operation_registry.update(_unary_ops)
operation_registry.update(_binary_infix_ops)
