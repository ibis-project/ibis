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

import sqlalchemy as sa

from ibis.sql.alchemy import unary, varargs, fixed_arity
import ibis.sql.alchemy as alch
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.common as com

_operation_registry = alch._operation_registry.copy()


def _cast(t, expr):
    # It's not all fun and games with SQLite

    op = expr.op()
    arg, target_type = op.args
    sa_arg = t.translate(arg)
    sa_type = t.get_sqla_type(target_type)

    if isinstance(target_type, dt.Timestamp):
        if not isinstance(arg, (ir.IntegerValue, ir.StringValue)):
            raise com.TranslationError(type(arg))

        return sa_arg

    if isinstance(arg, ir.CategoryValue) and target_type == 'int32':
        return sa_arg
    else:
        return sa.cast(sa_arg, sa_type)


def _substr(translator, expr):
    f = sa.func.substr

    arg, start, length = expr.op().args

    sa_arg = translator.translate(arg)
    sa_start = translator.translate(start)

    if length is None:
        return f(sa_arg, sa_start + 1)
    else:
        sa_length = translator.translate(length)
        return f(sa_arg, sa_start + 1, sa_length)


def _string_right(translator, expr):
    f = sa.func.substr

    arg, length = expr.op().args

    sa_arg = translator.translate(arg)
    sa_length = translator.translate(length)

    return f(sa_arg, -sa_length, sa_length)


def _string_find(translator, expr):
    arg, substr, start, _ = expr.op().args

    if start is not None:
        raise NotImplementedError

    sa_arg = translator.translate(arg)
    sa_substr = translator.translate(substr)

    f = sa.func.instr
    return f(sa_arg, sa_substr) - 1


def _infix_op(infix_sym):
    def formatter(translator, expr):
        op = expr.op()
        left, right = op.args

        left_arg = translator.translate(left)
        right_arg = translator.translate(right)
        return left_arg.op(infix_sym)(right_arg)

    return formatter


_operation_registry.update({
    ops.Cast: _cast,

    ops.Substring: _substr,
    ops.StrRight: _string_right,

    ops.StringFind: _string_find,

    ops.StringLength: unary('length'),

    ops.Least: varargs(sa.func.min),
    ops.Greatest: varargs(sa.func.max),
    ops.IfNull: fixed_arity(sa.func.ifnull, 2),

    ops.Lowercase: unary('lower'),
    ops.Uppercase: unary('upper'),

    ops.Strip: unary('trim'),
    ops.LStrip: unary('ltrim'),
    ops.RStrip: unary('rtrim'),

    ops.StringReplace: fixed_arity(sa.func.replace, 3),
    ops.StringSQLLike: _infix_op('LIKE'),
    ops.RegexSearch: _infix_op('REGEXP'),
})


def add_operation(op, translation_func):
    _operation_registry[op] = translation_func


class SQLiteExprTranslator(alch.AlchemyExprTranslator):

    _registry = _operation_registry
    _type_map = alch.AlchemyExprTranslator._type_map.copy()
    _type_map.update({
        dt.Double: sa.types.REAL
    })


class SQLiteDialect(alch.AlchemyDialect):

    translator = SQLiteExprTranslator
