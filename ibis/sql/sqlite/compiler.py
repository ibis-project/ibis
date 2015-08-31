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

_operation_registry = alch._operation_registry.copy()


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
