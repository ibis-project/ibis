# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# An Ibis analytical expression will typically consist of a primary SELECT
# statement, with zero or more supporting DDL queries. For example we would
# want to support converting a text file in HDFS to a Parquet-backed Impala
# table, with optional teardown if the user wants the intermediate converted
# table to be temporary.

import datetime
from io import BytesIO

import ibis
import ibis.expr.analytics as analytics
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.expr.temporal as tempo

import ibis.sql.transforms as transforms
import ibis.sql.identifiers as identifiers

import ibis.common as com
import ibis.util as util

# ---------------------------------------------------------------------
# Scalar and array expression formatting

_sql_type_names = {
    'int8': 'tinyint',
    'int16': 'smallint',
    'int32': 'int',
    'int64': 'bigint',
    'float': 'float',
    'double': 'double',
    'string': 'string',
    'boolean': 'boolean',
    'timestamp': 'timestamp'
}


def _cast(translator, expr):
    op = expr.op()
    arg = translator.translate(op.arg)
    if isinstance(op.arg, ir.CategoryValue) and op.target_type == 'int32':
        return arg
    else:
        sql_type = _type_to_sql_string(op.target_type)
        return 'CAST({0!s} AS {1!s})'.format(arg, sql_type)


def _type_to_sql_string(tval):
    if isinstance(tval, ir.DecimalType):
        return 'decimal({0},{1})'.format(tval.precision, tval.scale)
    else:
        return _sql_type_names[tval]


def _between(translator, expr):
    op = expr.op()
    comp = translator.translate(op.expr)
    lower = translator.translate(op.lower_bound)
    upper = translator.translate(op.upper_bound)
    return '{0!s} BETWEEN {1!s} AND {2!s}'.format(comp, lower, upper)


def _contains(translator, expr):
    op = expr.op()
    comp = translator.translate(op.value)
    options = translator.translate(op.options)
    return '{0!s} IN {1!s}'.format(comp, options)


def _like(translator, expr):
    op = expr.op()
    arg = translator.translate(op.arg)
    pattern = translator.translate(op.pattern)
    return '{0!s} LIKE {1!s}'.format(arg, pattern)


def _rlike(translator, expr):
    op = expr.op()
    arg = translator.translate(op.arg)
    pattern = translator.translate(op.pattern)
    return '{0!s} RLIKE {1!s}'.format(arg, pattern)


def _regex_extract(translator, expr):
    op = expr.op()
    formatted_arg = translator.translate(op.arg)
    formatted_pattern = translator.translate(op.pattern)
    return 'regexp_extract({0}, {1}, {2})'.format(formatted_arg,
                                                  formatted_pattern,
                                                  op.index)


def _regex_replace(translator, expr):
    op = expr.op()
    formatted_arg = translator.translate(op.arg)
    formatted_pattern = translator.translate(op.pattern)
    formatted_replacement = translator.translate(op.replacement)
    return 'regexp_replace({0}, {1}, {2})'.format(formatted_arg,
                                                  formatted_pattern,
                                                  formatted_replacement)


def _not_contains(translator, expr):
    # Slight code dup
    op = expr.op()
    comp = translator.translate(op.value)
    options = translator.translate(op.options)
    return '{0!s} NOT IN {1!s}'.format(comp, options)


def _is_null(translator, expr):
    formatted_arg = translator.translate(expr.op().arg)
    return '{0!s} IS NULL'.format(formatted_arg)


def _not_null(translator, expr):
    formatted_arg = translator.translate(expr.op().arg)
    return '{0!s} IS NOT NULL'.format(formatted_arg)


def _negate(translator, expr):
    arg = expr.op().arg
    formatted_arg = translator.translate(arg)
    if isinstance(expr, ir.BooleanValue):
        return 'NOT {0!s}'.format(formatted_arg)
    else:
        if _needs_parens(arg):
            formatted_arg = _parenthesize(formatted_arg)
        return '-{0!s}'.format(formatted_arg)


def _parenthesize(what):
    return '({0!s})'.format(what)


def _unary_op(func_name):
    def formatter(translator, expr):
        arg = translator.translate(expr.op().arg)
        return '{0!s}({1!s})'.format(func_name, arg)
    return formatter


def _reduction(func_name):
    def formatter(translator, expr):
        op = expr.op()

        if op.where is not None:
            case = op.where.ifelse(op.arg, ibis.NA)
            arg = translator.translate(case)
        else:
            arg = translator.translate(op.arg)

        return '{0!s}({1!s})'.format(func_name, arg)
    return formatter


def _fixed_arity_call(func_name, arity):
    def formatter(translator, expr):
        op = expr.op()
        formatted_args = []
        for i in xrange(arity):
            arg = op.args[i]
            fmt_arg = translator.translate(arg)
            formatted_args.append(fmt_arg)

        return '{0!s}({1!s})'.format(func_name, ', '.join(formatted_args))
    return formatter


def _binary_infix_op(infix_sym):
    def formatter(translator, expr):
        op = expr.op()

        left_arg = translator.translate(op.left)
        right_arg = translator.translate(op.right)

        if _needs_parens(op.left):
            left_arg = _parenthesize(left_arg)

        if _needs_parens(op.right):
            right_arg = _parenthesize(right_arg)

        return '{0!s} {1!s} {2!s}'.format(left_arg, infix_sym, right_arg)
    return formatter


def _xor(translator, expr):
    op = expr.op()

    left_arg = translator.translate(op.left)
    right_arg = translator.translate(op.right)

    if _needs_parens(op.left):
        left_arg = _parenthesize(left_arg)

    if _needs_parens(op.right):
        right_arg = _parenthesize(right_arg)

    return ('{0} AND NOT {1}'
            .format('({0} {1} {2})'.format(left_arg, 'OR', right_arg),
                    '({0} {1} {2})'.format(left_arg, 'AND', right_arg)))


def _name_expr(formatted_expr, quoted_name):
    return '{0!s} AS {1!s}'.format(formatted_expr, quoted_name)


def _needs_parens(op):
    if isinstance(op, ir.Expr):
        op = op.op()
    op_klass = type(op)
    # function calls don't need parens
    return (op_klass in _binary_infix_ops or
            op_klass in [ops.Negate])


def _need_parenthesize_args(op):
    if isinstance(op, ir.Expr):
        op = op.op()
    op_klass = type(op)
    return (op_klass in _binary_infix_ops or
            op_klass in [ops.Negate])


def _boolean_literal_format(expr):
    value = expr.op().value
    return 'TRUE' if value else 'FALSE'


def _number_literal_format(expr):
    value = expr.op().value
    return repr(value)


def _string_literal_format(expr):
    value = expr.op().value
    return "'{0!s}'".format(value.replace("'", "\\'"))


def _timestamp_literal_format(expr):
    value = expr.op().value
    if isinstance(value, datetime.datetime):
        if value.microsecond != 0:
            raise ValueError(value)
        value = value.strftime('%Y-%m-%d %H:%M:%S')

    return "'{0!s}'".format(value)


def quote_identifier(name, quotechar='`', force=False):
    if force or name.count(' ') or name in identifiers.impala_identifiers:
        return '{0}{1}{0}'.format(quotechar, name)
    else:
        return name


class CaseFormatter(object):

    def __init__(self, translator, base, cases, results, default):
        self.translator = translator
        self.base = base
        self.cases = cases
        self.results = results
        self.default = default

        # HACK
        self.indent = 2
        self.multiline = len(cases) > 1
        self.buf = BytesIO()

    def _trans(self, expr):
        return self.translator.translate(expr)

    def get_result(self):
        self.buf.seek(0)

        self.buf.write('CASE')
        if self.base is not None:
            base_str = self._trans(self.base)
            self.buf.write(' {0}'.format(base_str))

        for case, result in zip(self.cases, self.results):
            self._next_case()
            case_str = self._trans(case)
            result_str = self._trans(result)
            self.buf.write('WHEN {0} THEN {1}'.format(case_str, result_str))

        if self.default is not None:
            self._next_case()
            default_str = self._trans(self.default)
            self.buf.write('ELSE {0}'.format(default_str))

        if self.multiline:
            self.buf.write('\nEND')
        else:
            self.buf.write(' END')

        return self.buf.getvalue()

    def _next_case(self):
        if self.multiline:
            self.buf.write('\n{0}'.format(' ' * self.indent))
        else:
            self.buf.write(' ')


def _simple_case(translator, expr):
    op = expr.op()
    formatter = CaseFormatter(translator, op.base, op.cases, op.results,
                              op.default)
    return formatter.get_result()


def _searched_case(translator, expr):
    op = expr.op()
    formatter = CaseFormatter(translator, None, op.cases, op.results,
                              op.default)
    return formatter.get_result()


def _bucket(translator, expr):
    import operator

    op = expr.op()
    stmt = ibis.case()

    if op.closed == 'left':
        l_cmp = operator.le
        r_cmp = operator.lt
    else:
        l_cmp = operator.lt
        r_cmp = operator.le

    user_num_buckets = len(op.buckets) - 1

    bucket_id = 0
    if op.include_under:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else r_cmp
        else:
            cmp = operator.le if op.closed == 'right' else operator.lt
        stmt = stmt.when(cmp(op.arg, op.buckets[0]), bucket_id)
        bucket_id += 1

    for j, (lower, upper) in enumerate(zip(op.buckets, op.buckets[1:])):
        if (op.close_extreme
            and ((op.closed == 'right' and j == 0) or
                 (op.closed == 'left' and j == (user_num_buckets - 1)))):
            stmt = stmt.when((lower <= op.arg) & (op.arg <= upper),
                             bucket_id)
        else:
            stmt = stmt.when(l_cmp(lower, op.arg) & r_cmp(op.arg, upper),
                             bucket_id)
        bucket_id += 1

    if op.include_over:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else l_cmp
        else:
            cmp = operator.lt if op.closed == 'right' else operator.le

        stmt = stmt.when(cmp(op.buckets[-1], op.arg), bucket_id)
        bucket_id += 1

    case_expr = stmt.end().name(expr._name)
    return _searched_case(translator, case_expr)


def _category_label(translator, expr):
    op = expr.op()

    stmt = op.arg.case()
    for i, label in enumerate(op.labels):
        stmt = stmt.when(i, label)

    if op.nulls is not None:
        stmt = stmt.else_(op.nulls)

    case_expr = stmt.end().name(expr._name)
    return _simple_case(translator, case_expr)


def _table_array_view(translator, expr):
    ctx = translator.context
    table = expr.op().table
    query = ctx.get_formatted_query(table)
    return '(\n{0}\n)'.format(util.indent(query, ctx.indent))


# ---------------------------------------------------------------------
# Timestamp arithmetic and other functions

def _timestamp_delta(translator, expr):
    op = expr.op()
    formatted_arg = translator.translate(op.arg)
    return _timestamp_format_offset(op.offset, formatted_arg)


_impala_delta_functions = {
    tempo.Year: 'years_add',
    tempo.Month: 'months_add',
    tempo.Week: 'weeks_add',
    tempo.Day: 'days_add',
    tempo.Hour: 'hours_add',
    tempo.Minute: 'minutes_add',
    tempo.Second: 'seconds_add',
    tempo.Millisecond: 'milliseconds_add',
    tempo.Microsecond: 'microseconds_add',
    tempo.Nanosecond: 'nanoseconds_add'
}


def _timestamp_format_offset(offset, arg):
    f = _impala_delta_functions[type(offset)]
    return '{0}({1}, {2})'.format(f, arg, offset.n)


# ---------------------------------------------------------------------
# Semi/anti-join supports


def _any_exists(translator, expr):
    # Foreign references will have been catalogued by the correlated
    # ref-checking code. However, we need to rewrite this expression as a query
    # of the type
    #
    # SELECT 1
    # FROM {foreign_ref}
    # WHERE {correlated_filter}
    #
    # It's possible there could be multiple predicates inside the Any involving
    # more than one foreign reference. Will just disallow this for now until
    # someone *really* needs it.
    # op = expr.op()
    # ctx = translator.context

    # comp_op = op.arg.op()

    raise NotImplementedError


def _exists_subquery(translator, expr):
    op = expr.op()
    ctx = translator.context

    expr = (op.foreign_table
            .filter(op.predicates)
            .projection([ops.literal(1).name(ir.unnamed)]))

    subquery = ctx.get_formatted_query(expr)

    if isinstance(op, transforms.ExistsSubquery):
        key = 'EXISTS'
    elif isinstance(op, transforms.NotExistsSubquery):
        key = 'NOT EXISTS'
    else:
        raise NotImplementedError

    return '{0} (\n{1}\n)'.format(key, util.indent(subquery, ctx.indent))


def _table_column(translator, expr):
    op = expr.op()
    field_name = quote_identifier(op.name)

    table = op.table
    ctx = translator.context

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if translator.permit_subquery and ctx.is_foreign_expr(table):
        proj_expr = table.projection([field_name]).to_array()
        return _table_array_view(translator, proj_expr)

    if ctx.need_aliases():
        alias = ctx.get_alias(table)
        if alias is not None:
            field_name = '{0}.{1}'.format(alias, field_name)

    return field_name


def _extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.arg)

        # This is pre-2.0 Impala-style, which did not used to support the
        # SQL-99 format extract($FIELD from expr)
        return "extract({0!s}, '{1!s}')".format(arg, sql_attr)
    return extract_field_formatter


def _timestamp_from_unix(translator, expr):
    op = expr.op()

    val = op.arg
    if op.unit == 'ms':
        val = (val / 1000).cast('int32')
    elif op.unit == 'us':
        val = (val / 1000000).cast('int32')

    arg = _from_unixtime(translator, val)
    return 'CAST({0} AS timestamp)'.format(arg)


def _from_unixtime(translator, expr):
    arg = translator.translate(expr)
    return 'from_unixtime({0}, "yyyy-MM-dd HH:mm:ss")'.format(arg)


def _coalesce_like(func_name):
    def coalesce_like_formatter(translator, expr):
        op = expr.op()
        trans_args = [translator.translate(arg) for arg in op.args]
        return '{0}({1})'.format(func_name, ', '.join(trans_args))
    return coalesce_like_formatter


def _substring(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)

    # Databases are 1-indexed
    if op.length:
        return 'substr({0}, {1}, {2})'.format(arg_formatted, op.start + 1,
                                              op.length)
    else:
        return 'substr({0}, {1})'.format(arg_formatted, op.start + 1)


def _strright(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)
    return 'strright({0}, {1})'.format(arg_formatted, op.nchars)


def _repeat(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)
    return 'repeat({0}, {1})'.format(arg_formatted, op.n)


def _string_find(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)
    substr_formatted = translator.translate(op.substr)
    return 'instr({0}, {1}) - 1'.format(arg_formatted, substr_formatted)


def _locate(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)
    substr_formatted = translator.translate(op.substr)

    if op.pos:
        return 'locate({0}, {1}, {2}) - 1'.format(substr_formatted,
                                                  arg_formatted,
                                                  op.pos + 1)
    else:
        return 'locate({0}, {1}) - 1'.format(substr_formatted, arg_formatted)


def _string_join(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)
    strings_formatted = [translator.translate(x) for x in op.strings]
    return 'concat_ws({0}, {1})'.format(arg_formatted,
                                        ', '.join(strings_formatted))


def _find_in_set(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)
    str_formatted = ','.join([x._arg.value for x in op.str_list])
    return "find_in_set({0}, '{1}') - 1".format(arg_formatted, str_formatted)


def _round(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)

    if op.digits is not None:
        return 'round({0}, {1})'.format(arg_formatted, op.digits)
    else:
        return 'round({0})'.format(arg_formatted)


def _hash(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)

    if op.how == 'fnv':
        return 'fnv_hash({0})'.format(arg_formatted)
    else:
        raise NotImplementedError(op.how)


def _log(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)

    if op.base is None:
        return 'ln({0})'.format(arg_formatted)
    else:
        return 'log({0}, {1})'.format(arg_formatted, op.base)


def _count_distinct(translator, expr):
    op = expr.op()
    arg_formatted = translator.translate(op.arg)
    return 'COUNT(DISTINCT {0})'.format(arg_formatted)


def _literal(translator, expr):
    if isinstance(expr, ir.BooleanValue):
        typeclass = 'boolean'
    elif isinstance(expr, ir.StringValue):
        typeclass = 'string'
    elif isinstance(expr, ir.NumericValue):
        typeclass = 'number'
    elif isinstance(expr, ir.TimestampValue):
        typeclass = 'timestamp'
    else:
        raise NotImplementedError

    return _literal_formatters[typeclass](expr)


def _null_literal(translator, expr):
    return 'NULL'


_literal_formatters = {
    'boolean': _boolean_literal_format,
    'number': _number_literal_format,
    'string': _string_literal_format,
    'timestamp': _timestamp_literal_format
}


def _value_list(translator, expr):
    op = expr.op()
    formatted = [translator.translate(x) for x in op.values]
    return '({0})'.format(', '.join(formatted))


def _not_implemented(translator, expr):
    raise NotImplementedError


_unary_ops = {
    # Unary operations
    ops.NotNull: _not_null,
    ops.IsNull: _is_null,
    ops.Negate: _negate,

    ops.IfNull: _fixed_arity_call('isnull', 2),
    ops.NullIf: _fixed_arity_call('nullif', 2),

    ops.ZeroIfNull: _unary_op('zeroifnull'),

    ops.Abs: _unary_op('abs'),
    ops.Ceil: _unary_op('ceil'),
    ops.Floor: _unary_op('floor'),
    ops.Exp: _unary_op('exp'),
    ops.Round: _round,

    ops.Sign: _unary_op('sign'),
    ops.Sqrt: _unary_op('sqrt'),

    ops.Hash: _hash,

    ops.Log: _log,
    ops.Ln: _unary_op('ln'),
    ops.Log2: _unary_op('log2'),
    ops.Log10: _unary_op('log10'),

    ops.DecimalPrecision: _unary_op('precision'),
    ops.DecimalScale: _unary_op('scale'),

    # Unary aggregates
    ops.CMSMedian: _reduction('appx_median'),
    ops.HLLCardinality: _reduction('ndv'),
    ops.Mean: _reduction('avg'),
    ops.Sum: _reduction('sum'),
    ops.Max: _reduction('max'),
    ops.Min: _reduction('min'),
    ops.GroupConcat: _fixed_arity_call('group_concat', 2),

    ops.Count: _reduction('count'),
    ops.CountDistinct: _count_distinct,
}


_binary_infix_ops = {
    # Binary operations
    ops.Add: _binary_infix_op('+'),
    ops.Subtract: _binary_infix_op('-'),
    ops.Multiply: _binary_infix_op('*'),
    ops.Divide: _binary_infix_op('/'),
    ops.Power: _fixed_arity_call('pow', 2),
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

_string_ops = {
    ops.StringLength: _unary_op('length'),
    ops.StringAscii: _unary_op('ascii'),
    ops.Lowercase: _unary_op('lower'),
    ops.Uppercase: _unary_op('upper'),
    ops.Reverse: _unary_op('reverse'),
    ops.Trim: _unary_op('trim'),
    ops.LTrim: _unary_op('ltrim'),
    ops.RTrim: _unary_op('rtrim'),
    ops.Substring: _substring,
    ops.StrRight: _fixed_arity_call('strright', 2),
    ops.Repeat: _fixed_arity_call('repeat', 2),
    ops.StringFind: _string_find,
    ops.Translate: _fixed_arity_call('translate', 3),
    ops.FindInSet: _find_in_set,
    ops.LPad: _fixed_arity_call('lpad', 3),
    ops.RPad: _fixed_arity_call('rpad', 3),
    ops.Locate: _locate,
    ops.StringJoin: _string_join,
    ops.StringSQLLike: _like,
    ops.RegexSearch: _rlike,
    ops.RegexExtract: _regex_extract,
    ops.RegexReplace: _regex_replace,
}


_timestamp_ops = {
    ops.TimestampNow: lambda *args: 'now()',
    ops.ExtractYear: _extract_field('year'),
    ops.ExtractMonth: _extract_field('month'),
    ops.ExtractDay: _extract_field('day'),
    ops.ExtractHour: _extract_field('hour'),
    ops.ExtractMinute: _extract_field('minute'),
    ops.ExtractSecond: _extract_field('second'),
    ops.ExtractMillisecond: _extract_field('millisecond'),
}


_other_ops = {
    ops.Any: _any_exists,

    ops.E: lambda *args: 'e()',

    ir.Literal: _literal,
    ops.NullLiteral: _null_literal,

    ops.ValueList: _value_list,

    ops.Cast: _cast,

    ops.Coalesce: _coalesce_like('coalesce'),
    ops.Greatest: _coalesce_like('greatest'),
    ops.Least: _coalesce_like('least'),

    ops.Where: _fixed_arity_call('if', 3),

    ops.Between: _between,
    ops.Contains: _contains,
    ops.NotContains: _not_contains,

    analytics.Bucket: _bucket,
    analytics.CategoryLabel: _category_label,

    ops.SimpleCase: _simple_case,
    ops.SearchedCase: _searched_case,

    ops.TableColumn: _table_column,

    ops.TableArrayView: _table_array_view,

    ops.TimestampDelta: _timestamp_delta,
    ops.TimestampFromUNIX: _timestamp_from_unix,

    transforms.ExistsSubquery: _exists_subquery,
    transforms.NotExistsSubquery: _exists_subquery
}


_operation_registry = {}
_operation_registry.update(_unary_ops)
_operation_registry.update(_binary_infix_ops)
_operation_registry.update(_string_ops)
_operation_registry.update(_timestamp_ops)
_operation_registry.update(_other_ops)


class ExprTranslator(object):

    def __init__(self, expr, context=None, named=False, permit_subquery=False):
        self.expr = expr
        self.permit_subquery = permit_subquery

        if context is None:
            from ibis.sql.compiler import QueryContext
            context = QueryContext()
        self.context = context

        # For now, governing whether the result will have a name
        self.named = named

    def get_result(self):
        """
        Build compiled SQL expression from the bottom up and return as a string
        """
        translated = self.translate(self.expr)
        if self._needs_name(self.expr):
            # TODO: this could fail in various ways
            name = self.expr.get_name()
            translated = _name_expr(translated,
                                    quote_identifier(name, force=True))
        return translated

    def _needs_name(self, expr):
        if not self.named:
            return False

        op = expr.op()
        if isinstance(op, ops.TableColumn):
            # This column has been given an explicitly different name
            if expr.get_name() != op.name:
                return True
            return False

        if expr.get_name() is ir.unnamed:
            return False

        return True

    def translate(self, expr):
        # The operation node type the typed expression wraps
        op = expr.op()

        # TODO: use op MRO for subclasses instead of this isinstance spaghetti
        if isinstance(op, ir.Parameter):
            return self._trans_param(expr)
        elif isinstance(op, ops.TableNode):
            # HACK/TODO: revisit for more complex cases
            return '*'
        elif type(op) in _operation_registry:
            formatter = _operation_registry[type(op)]
            return formatter(self, expr)
        else:
            raise com.TranslationError('No translator rule for {0}'.format(op))

    def _trans_param(self, expr):
        raise NotImplementedError
