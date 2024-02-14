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

from __future__ import annotations

import contextlib

from sqlglot.dialects import DuckDB

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend
from ibis.expr.schema import Schema
from ibis.expr.tests.conftest import MOCK_TABLES


class MockBackend(BaseBackend):
    name = "mock"
    version = "1.0"
    current_database = "mockdb"
    dialect = DuckDB

    def __init__(self):
        super().__init__()
        self.executed_queries = []
        self.sql_query_schemas = {}

    def do_connect(self):
        pass

    def disconnect(self):
        pass

    def table(self, name, **kwargs):
        schema = self.get_schema(name)
        node = ops.DatabaseTable(source=self, name=name, schema=schema)
        return node.to_expr()

    def list_tables(self):
        return list(MOCK_TABLES)

    def list_databases(self):
        return ["mockdb"]

    def _to_sql(self, expr, **kwargs):
        import ibis

        return ibis.to_sql(expr, dialect="duckdb", **kwargs)

    def fetch_from_cursor(self, cursor, schema):
        pass

    def get_schema(self, name):
        name = name.replace("`", "")
        return Schema.from_tuples(MOCK_TABLES[name])

    def to_pyarrow(self, *_, **__):
        raise NotImplementedError(self.name)

    def execute(self, *_, **__):
        raise NotImplementedError(self.name)

    def compile(self, *_, **__):
        raise NotImplementedError(self.name)

    def create_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def create_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def _load_into_cache(self, *_):
        raise NotImplementedError(self.name)

    def _clean_up_cached_table(self, _):
        raise NotImplementedError(self.name)

    def _get_schema_using_query(self, query):
        return self.sql_query_schemas[query]

    def _get_sql_string_view_schema(self, name, table, query):
        return self.sql_query_schemas[query]

    @contextlib.contextmanager
    def set_query_schema(self, query, schema):
        self.sql_query_schemas[query] = schema
        yield
        self.sql_query_schemas.pop(query, None)
