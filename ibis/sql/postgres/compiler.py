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

import re
import locale
import string

from functools import reduce

import sqlalchemy as sa

from ibis.sql.alchemy import unary, varargs, fixed_arity
import ibis.sql.alchemy as alch
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


_operation_registry = alch._operation_registry.copy()


# TODO: substr and find are copied from SQLite, we should really have a
# "base" set of SQL functions that are the most common APIs across the major
# RDBMS
def _substr(t, expr):
    f = sa.func.substr

    arg, start, length = expr.op().args

    sa_arg = t.translate(arg)
    sa_start = t.translate(start)

    if length is None:
        return f(sa_arg, sa_start + 1)
    else:
        sa_length = t.translate(length)
        return f(sa_arg, sa_start + 1, sa_length)


def _string_find(t, expr):
    arg, substr, start, _ = expr.op().args

    if start is not None:
        raise NotImplementedError

    sa_arg = t.translate(arg)
    sa_substr = t.translate(substr)

    return sa.func.strpos(sa_arg, sa_substr) - 1


def _infix_op(infix_sym):
    def formatter(t, expr):
        op = expr.op()
        left, right = op.args

        left_arg = t.translate(left)
        right_arg = t.translate(right)
        return left_arg.op(infix_sym)(right_arg)

    return formatter


def _extract(fmt):
    def translator(t, expr):
        arg, = expr.op().args
        sa_arg = t.translate(arg)
        return sa.extract(fmt, sa_arg)
    return translator


def _cast(t, expr):
    arg, typ = expr.op().args

    sa_arg = t.translate(arg)
    sa_type = t.get_sqla_type(typ)

    # specialize going from an integer type to a timestamp
    if isinstance(arg.type(), dt.Integer) and issubclass(sa_type, sa.DateTime):
        return sa.func.timezone('UTC', sa.func.to_timestamp(sa_arg))
    return sa.cast(sa_arg, sa_type)


def _typeof(t, expr):
    arg, = expr.op().args
    sa_arg = t.translate(arg)
    typ = sa.cast(sa.func.pg_typeof(sa_arg), sa.TEXT)

    # select pg_typeof('asdf') returns unknown so we have to check the child's
    # type for nullness
    return sa.case(
        [
            ((typ == 'unknown') & (arg.type() != dt.null), 'text'),
            ((typ == 'unknown') & (arg.type() == dt.null), 'null'),
        ],
        else_=typ
    )


def _second(t, expr):
    # extracting the second gives us the fractional part as well, so smash that
    # with a cast to SMALLINT
    sa_arg, = map(t.translate, expr.op().args)
    return sa.cast(sa.extract('second', sa_arg), sa.SMALLINT)


def _millisecond(t, expr):
    # we get total number of milliseconds including seconds with extract so we
    # mod 1000
    sa_arg, = map(t.translate, expr.op().args)
    return sa.cast(sa.extract('millisecond', sa_arg), sa.SMALLINT) % 1000


def _string_agg(t, expr):
    # we could use sa.func.string_agg since postgres 9.0, but we can cheaply
    # maintain backwards compatibility here, so we don't use it
    arg, sep = map(t.translate, expr.op().args)
    return sa.func.array_to_string(sa.func.array_agg(arg), sep)


_strftime_to_postgresql_rules = {
    '%a': 'TMDy',  # TM does it in a locale dependent way
    '%A': 'TMDay',
    '%w': 'D',  # 1-based day of week, see below for how we make this 0-based
    '%d': 'DD',  # day of month
    '%-d': 'FMDD',  # - is no leading zero for Python same for FM in postgres
    '%b': 'TMMon',  # Sep
    '%B': 'TMMonth',  # September
    '%m': 'MM',  # 01
    '%-m': 'FMMM',  # 1
    '%y': 'YY',  # 15
    '%Y': 'YYYY',  # 2015
    '%H': 'HH24',  # 09
    '%-H': 'FMHH24',  # 9
    '%I': 'HH12',  # 09
    '%-I': 'FMHH12',  # 9
    '%p': 'AM',  # AM or PM
    '%M': 'MI',  # zero padded minute
    '%-M': 'FMMI',  # Minute
    '%S': 'SS',  # zero padded second
    '%-S': 'FMSS',  # Second
    '%f': 'US',  # zero padded microsecond
    '%z': 'OF',  # utf offset
    '%Z': 'TZ',  # uppercase timezone name
    '%j': 'DDD',  # zero padded day of year
    '%-j': 'FMDDD',  # day of year
    '%U': 'WW',  # 1-based week of year
    # 'W': ?,  # meh
    '%c': locale.nl_langinfo(locale.D_T_FMT),  # locale date and time
    '%x': locale.nl_langinfo(locale.D_FMT),  # locale date
    '%X': locale.nl_langinfo(locale.T_FMT)  # locale time
}


def tokenize_noop(scanner, token):
    return token


# translate strftime spec into mostly equivalent PostgreSQL spec
_scanner = re.Scanner([
    (py, tokenize_noop)
    for py in _strftime_to_postgresql_rules.keys()
] + [
    # "%e" is in the C standard and Python actually generates this if your spec
    # contains "%c" but we don't officially support it as a specifier so we
    # need to special case it in the scanner
    (r'%e', tokenize_noop),

    # double quotes need to be escaped
    (r'"', lambda scanner, token: re.escape(token)),

    # spaces should be greedily consumed and kept
    (r'\s+', tokenize_noop),

    (r'[%s]' % re.escape(string.punctuation), tokenize_noop),

    # everything else except double quotes and spaces
    (r'[^%s\s]+' % re.escape(string.punctuation), tokenize_noop),
])


_lexicon_values = frozenset(_strftime_to_postgresql_rules.values())

_strftime_blacklist = frozenset(['%w', '%U', '%c', '%x', '%X', '%e'])


def _reduce_tokens(tokens, arg):
    # current list of tokens
    curtokens = []

    # reduced list of tokens that accounts for blacklisted values
    reduced = []

    non_special_tokens = (
        frozenset(_strftime_to_postgresql_rules) - _strftime_blacklist
    )

    # TODO: how much of a hack is this?
    for token in tokens:

        # we are a non-special token %A, %d, etc.
        if token in non_special_tokens:
            curtokens.append(_strftime_to_postgresql_rules[token])

        # we have a string like DD, to escape this we
        # surround it with double quotes
        elif token in _lexicon_values:
            curtokens.append('"%s"' % token)

        # we have a token that needs special treatment
        elif token in _strftime_blacklist:
            if token == '%w':
                value = sa.extract('dow', arg)  # 0 based day of week
            elif token == '%U':
                value = sa.cast(sa.func.to_char(arg, 'WW'), sa.SMALLINT) - 1
            elif token == '%c' or token == '%x' or token == '%X':
                # re scan and tokenize this pattern
                new_tokens, _ = _scanner.scan(
                    _strftime_to_postgresql_rules[token]
                )
                value = reduce(
                    sa.sql.ColumnElement.concat,
                    _reduce_tokens(new_tokens, arg)
                )
            elif token == '%e':
                # pad with spaces instead of zeros
                value = sa.func.replace(sa.func.to_char(arg, 'DD'), '0', ' ')

            reduced += [
                sa.func.to_char(arg, ''.join(curtokens)),
                sa.cast(value, sa.TEXT)
            ]

            # empty current token list in case there are more tokens
            del curtokens[:]

        # uninteresting text
        else:
            curtokens.append(token)
    else:
        # append result to r if we had more tokens or if we have no
        # blacklisted tokens
        if curtokens:
            reduced.append(sa.func.to_char(arg, ''.join(curtokens)))
    return reduced


def _strftime(t, expr):
    arg, pattern = map(t.translate, expr.op().args)
    tokens, _ = _scanner.scan(pattern.value)
    reduced = _reduce_tokens(tokens, arg)
    return reduce(sa.sql.ColumnElement.concat, reduced)


def _find_in_set(t, expr):
    # postgresql 9.5 has array_position, but the code below works on any
    # version with generate_subscripts
    # TODO: could make it even more generic by not using generate_subscripts
    # TODO: this works with *any* type, not just strings. should the operation
    #       itself also have this property?
    arg, haystack = expr.op().args
    needle = t.translate(arg)
    haystack = sa.select([sa.literal(
        [element._arg.value for element in haystack],
        type_=sa.dialects.postgresql.ARRAY(needle.type)
    ).label('haystack')]).alias()
    subscripts = sa.select([
        sa.func.generate_subscripts(haystack.c.haystack, 1).label('i')
    ]).alias()

    # return a zero based index
    result = sa.select([subscripts.c.i - 1]).where(
        haystack.c.haystack[subscripts.c.i] == needle
    ).order_by(subscripts.c.i).limit(1)
    return result


def _regex_replace(t, expr):
    string, pattern, replacement = map(t.translate, expr.op().args)

    # postgres defaults to replacing only the first occurrence
    return sa.func.regexp_replace(string, pattern, replacement, 'g')


_operation_registry.update({
    # types
    ops.Cast: _cast,
    ops.TypeOf: _typeof,

    # miscellaneous varargs
    ops.Least: varargs(sa.func.least),
    ops.Greatest: varargs(sa.func.greatest),

    # null handling
    ops.IfNull: fixed_arity(sa.func.coalesce, 2),
    ops.Coalesce: varargs(sa.func.coalesce),

    # boolean reductions
    ops.Any: fixed_arity(sa.func.bool_or, 1),
    ops.All: fixed_arity(sa.func.bool_and, 1),
    ops.NotAny: fixed_arity(lambda x: sa.not_(sa.func.bool_or(x)), 1),
    ops.NotAll: fixed_arity(lambda x: sa.not_(sa.func.bool_and(x)), 1),

    # strings
    ops.Substring: _substr,
    ops.StrRight: fixed_arity(sa.func.right, 2),
    ops.StringFind: _string_find,
    ops.StringLength: unary('length'),
    ops.GroupConcat: _string_agg,
    ops.Lowercase: unary('lower'),
    ops.Uppercase: unary('upper'),
    ops.Strip: unary('trim'),
    ops.LStrip: unary('ltrim'),
    ops.RStrip: unary('rtrim'),
    ops.LPad: fixed_arity('lpad', 3),
    ops.RPad: fixed_arity('rpad', 3),
    ops.Reverse: unary('reverse'),
    ops.Capitalize: unary('initcap'),
    ops.Repeat: fixed_arity('repeat', 2),
    ops.StringReplace: fixed_arity(sa.func.replace, 3),
    ops.StringSQLLike: _infix_op('LIKE'),
    ops.RegexSearch: _infix_op('~'),
    ops.RegexReplace: _regex_replace,
    ops.Translate: fixed_arity('translate', 3),
    ops.FindInSet: _find_in_set,
    # ops.RegexExtract: ...,

    # dates and times
    ops.Strftime: _strftime,
    ops.ExtractYear: _extract('year'),
    ops.ExtractMonth: _extract('month'),
    ops.ExtractDay: _extract('day'),
    ops.ExtractHour: _extract('hour'),
    ops.ExtractMinute: _extract('minute'),
    ops.ExtractSecond: _second,
    ops.ExtractMillisecond: _millisecond,

    # now is in the timezone of the server, but we want UTC
    ops.TimestampNow: lambda *args: sa.func.timezone('UTC', sa.func.now()),
})


def add_operation(op, translation_func):
    _operation_registry[op] = translation_func


class PostgreSQLExprTranslator(alch.AlchemyExprTranslator):

    _registry = _operation_registry
    _rewrites = alch.AlchemyExprTranslator._rewrites.copy()
    _type_map = alch.AlchemyExprTranslator._type_map.copy()
    _type_map.update({
        dt.Double: sa.types.FLOAT,
        dt.Float: sa.types.REAL
    })


rewrites = PostgreSQLExprTranslator.rewrites
compiles = PostgreSQLExprTranslator.compiles


class PostgreSQLDialect(alch.AlchemyDialect):

    translator = PostgreSQLExprTranslator
