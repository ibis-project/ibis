from __future__ import annotations

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

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.sqlite.registry import operation_registry


class SQLiteExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Float64: sa.types.REAL,
            dt.Float16: sa.types.REAL,
            dt.Float32: sa.types.REAL,
        }
    )


rewrites = SQLiteExprTranslator.rewrites


@rewrites(ops.DayOfWeekIndex)
def day_of_week_index(op):
    # TODO(kszucs): avoid expr roundtrip
    expr = op.arg.to_expr()
    new_expr = ((expr.strftime('%w').cast(dt.int16) + 6) % 7).cast(dt.int16)
    return new_expr.op()


@rewrites(ops.DayOfWeekName)
def day_of_week_name(op):
    # TODO(kszucs): avoid expr roundtrip
    expr = op.arg.to_expr()
    new_expr = (
        expr.day_of_week.index()
        .case()
        .when(0, 'Monday')
        .when(1, 'Tuesday')
        .when(2, 'Wednesday')
        .when(3, 'Thursday')
        .when(4, 'Friday')
        .when(5, 'Saturday')
        .when(6, 'Sunday')
        .else_(ibis.NA)
        .end()
    )
    return new_expr.op()


class SQLiteCompiler(AlchemyCompiler):
    translator_class = SQLiteExprTranslator
