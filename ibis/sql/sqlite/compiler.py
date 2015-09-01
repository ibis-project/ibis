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

import ibis.sql.alchemy as alch
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

_operation_registry = alch._operation_registry.copy()


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


def _unary_op(sa_func):
    return alch._fixed_arity_call(sa_func, 1)


_operation_registry.update({
    ops.Substring: _substr,
    ops.StrRight: _string_right,

    ops.StringLength: _unary_op('length'),

    ops.Lowercase: _unary_op('lower'),
    ops.Uppercase: _unary_op('upper'),

    ops.Strip: _unary_op('trim'),
    ops.LStrip: _unary_op('ltrim'),
    ops.RStrip: _unary_op('rtrim'),
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
