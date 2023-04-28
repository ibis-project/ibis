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
from sqlalchemy.dialects import sqlite

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    to_sqla_type,
)
from ibis.backends.sqlite.registry import operation_registry


class SQLiteExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _dialect_name = "sqlite"


rewrites = SQLiteExprTranslator.rewrites


class SQLiteCompiler(AlchemyCompiler):
    translator_class = SQLiteExprTranslator
    support_values_syntax_in_select = False


@to_sqla_type.register(sqlite.dialect, (dt.Float32, dt.Float64))
def _floating_point(_, itype):
    return sa.REAL
