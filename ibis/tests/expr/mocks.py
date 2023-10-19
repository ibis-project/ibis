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

import pytest
import sqlalchemy as sa

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.base.sql.alchemy import AlchemyCompiler
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.expr.schema import Schema
from ibis.expr.tests.conftest import MOCK_TABLES


class MockBackend(BaseSQLBackend):
    name = "mock"
    version = "1.0"
    current_database = "mockdb"

    def __init__(self):
        super().__init__()
        self.executed_queries = []
        self.sql_query_schemas = {}

    def do_connect(self):
        pass

    def list_tables(self):
        return list(MOCK_TABLES)

    def list_databases(self):
        return ["mockdb"]

    def fetch_from_cursor(self, cursor, schema):
        pass

    def get_schema(self, name):
        name = name.replace("`", "")
        return Schema.from_tuples(MOCK_TABLES[name])

    def to_pyarrow(self, expr, limit=None, params=None, **kwargs):
        ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        for query in ast.queries:
            self.executed_queries.append(query.compile())

        if isinstance(expr, ir.Scalar):
            return None
        elif isinstance(expr, ir.Column):
            schema = expr.as_table().schema()
            return schema.to_pyarrow().empty_table()[0]
        else:
            return expr.schema().to_pyarrow().empty_table()

    def execute(self, expr, limit=None, params=None, **kwargs):
        out = self.to_pyarrow(expr, limit=limit, params=params, **kwargs)
        return None if out is None else out.to_pandas()

    def compile(
        self,
        expr,
        limit=None,
        params=None,
        timecontext=None,
    ):
        ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        queries = [q.compile() for q in ast.queries]
        return queries[0] if len(queries) == 1 else queries

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

    @contextlib.contextmanager
    def set_query_schema(self, query, schema):
        self.sql_query_schemas[query] = schema
        yield
        self.sql_query_schemas.pop(query, None)


def table_from_schema(name, meta, schema, *, database: str | None = None):
    # Convert Ibis schema to SQLA table
    columns = []

    for colname, dtype in zip(schema.names, schema.types):
        satype = AlchemyType.from_ibis(dtype)
        column = sa.Column(colname, satype, nullable=dtype.nullable)
        columns.append(column)

    return sa.Table(name, meta, *columns, schema=database)


class MockAlchemyBackend(MockBackend):
    compiler = AlchemyCompiler

    def __init__(self):
        super().__init__()
        pytest.importorskip("sqlalchemy")
        self.tables = {}

    def table(self, name, **_):
        schema = self.get_schema(name)
        return self._inject_table(name, schema)

    def _inject_table(self, name, schema):
        if name not in self.tables:
            self.tables[name] = table_from_schema(name, sa.MetaData(), schema)
        return ops.DatabaseTable(source=self, name=name, schema=schema).to_expr()

    def _get_sqla_table(self, name, **_):
        return self.tables[name]


GEO_TABLE = {
    "geo": [
        ("id", "int32"),
        ("geo_point", "point"),
        ("geo_linestring", "linestring"),
        ("geo_polygon", "polygon"),
        ("geo_multipolygon", "multipolygon"),
    ]
}


class GeoMockConnectionPostGIS(MockAlchemyBackend):
    _tables = GEO_TABLE

    def __init__(self):
        super().__init__()
        self.executed_queries = []

    def get_schema(self, name):
        return Schema.from_tuples(self._tables[name])


class GeoMockConnectionOmniSciDB(MockBackend):
    _tables = GEO_TABLE

    def __init__(self):
        super().__init__()
        self.executed_queries = []

    def get_schema(self, name):
        return Schema.from_tuples(self._tables[name])
