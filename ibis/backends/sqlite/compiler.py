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
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.sqlite.datatypes import SqliteType
from ibis.backends.sqlite.registry import operation_registry
from ibis.expr.rewrites import rewrite_sample


class SQLiteExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _dialect_name = "sqlite"
    type_mapper = SqliteType


rewrites = SQLiteExprTranslator.rewrites


class SQLiteCompiler(AlchemyCompiler):
    translator_class = SQLiteExprTranslator
    support_values_syntax_in_select = False
    null_limit = None
    rewrites = AlchemyCompiler.rewrites | rewrite_sample
