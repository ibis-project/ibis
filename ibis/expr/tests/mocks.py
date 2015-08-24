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

from ibis.client import SQLClient
from ibis.expr.datatypes import Schema
import ibis


class MockConnection(SQLClient):

    _tables = {
        'alltypes': [
            ('a', 'int8'),
            ('b', 'int16'),
            ('c', 'int32'),
            ('d', 'int64'),
            ('e', 'float'),
            ('f', 'double'),
            ('g', 'string'),
            ('h', 'boolean'),
            ('i', 'timestamp')
        ],
        'star1': [
            ('c', 'int32'),
            ('f', 'double'),
            ('foo_id', 'string'),
            ('bar_id', 'string'),
        ],
        'star2': [
            ('foo_id', 'string'),
            ('value1', 'double'),
            ('value3', 'double')
        ],
        'star3': [
            ('bar_id', 'string'),
            ('value2', 'double')
        ],
        'test1': [
            ('c', 'int32'),
            ('f', 'double'),
            ('g', 'string')
        ],
        'test2': [
            ('key', 'string'),
            ('value', 'double')
        ],
        'tpch_region': [
            ('r_regionkey', 'int16'),
            ('r_name', 'string'),
            ('r_comment', 'string')
        ],
        'tpch_nation': [
            ('n_nationkey', 'int16'),
            ('n_name', 'string'),
            ('n_regionkey', 'int16'),
            ('n_comment', 'string')
        ],
        'tpch_lineitem': [
            ('l_orderkey', 'int64'),
            ('l_partkey', 'int64'),
            ('l_suppkey', 'int64'),
            ('l_linenumber', 'int32'),
            ('l_quantity', 'decimal(12,2)'),
            ('l_extendedprice', 'decimal(12,2)'),
            ('l_discount', 'decimal(12,2)'),
            ('l_tax', 'decimal(12,2)'),
            ('l_returnflag', 'string'),
            ('l_linestatus', 'string'),
            ('l_shipdate', 'string'),
            ('l_commitdate', 'string'),
            ('l_receiptdate', 'string'),
            ('l_shipinstruct', 'string'),
            ('l_shipmode', 'string'),
            ('l_comment', 'string')
        ],
        'tpch_customer': [
            ('c_custkey', 'int64'),
            ('c_name', 'string'),
            ('c_address', 'string'),
            ('c_nationkey', 'int16'),
            ('c_phone', 'string'),
            ('c_acctbal', 'decimal'),
            ('c_mktsegment', 'string'),
            ('c_comment', 'string')
        ],
        'tpch_orders': [
            ('o_orderkey', 'int64'),
            ('o_custkey', 'int64'),
            ('o_orderstatus', 'string'),
            ('o_totalprice', 'decimal(12,2)'),
            ('o_orderdate', 'string'),
            ('o_orderpriority', 'string'),
            ('o_clerk', 'string'),
            ('o_shippriority', 'int32'),
            ('o_comment', 'string')
        ],
        'functional_alltypes': [
            ('id', 'int32'),
            ('bool_col', 'boolean'),
            ('tinyint_col', 'int8'),
            ('smallint_col', 'int16'),
            ('int_col', 'int32'),
            ('bigint_col', 'int64'),
            ('float_col', 'float'),
            ('double_col', 'double'),
            ('date_string_col', 'string'),
            ('string_col', 'string'),
            ('timestamp_col', 'timestamp'),
            ('year', 'int32'),
            ('month', 'int32')
        ],
        'airlines': [
            ('year', 'int32'),
            ('month', 'int32'),
            ('day', 'int32'),
            ('dayofweek', 'int32'),
            ('dep_time', 'int32'),
            ('crs_dep_time', 'int32'),
            ('arr_time', 'int32'),
            ('crs_arr_time', 'int32'),
            ('carrier', 'string'),
            ('flight_num', 'int32'),
            ('tail_num', 'int32'),
            ('actual_elapsed_time', 'int32'),
            ('crs_elapsed_time', 'int32'),
            ('airtime', 'int32'),
            ('arrdelay', 'int32'),
            ('depdelay', 'int32'),
            ('origin', 'string'),
            ('dest', 'string'),
            ('distance', 'int32'),
            ('taxi_in', 'int32'),
            ('taxi_out', 'int32'),
            ('cancelled', 'int32'),
            ('cancellation_code', 'string'),
            ('diverted', 'int32'),
            ('carrier_delay', 'int32'),
            ('weather_delay', 'int32'),
            ('nas_delay', 'int32'),
            ('security_delay', 'int32'),
            ('late_aircraft_delay', 'int32')
        ],
    }

    def __init__(self):
        self.executed_queries = []

    def _get_table_schema(self, name):
        name = name.replace('`', '')
        return Schema.from_tuples(self._tables[name])

    def execute(self, expr, limit=None):
        ast = self._build_ast_ensure_limit(expr, limit)
        for query in ast.queries:
            self.executed_queries.append(query.compile())
        return None


_all_types_schema = [
    ('a', 'int8'),
    ('b', 'int16'),
    ('c', 'int32'),
    ('d', 'int64'),
    ('e', 'float'),
    ('f', 'double'),
    ('g', 'string'),
    ('h', 'boolean')
]


class BasicTestCase(object):

    def setUp(self):
        self.schema = _all_types_schema
        self.schema_dict = dict(self.schema)
        self.table = ibis.table(self.schema)

        self.int_cols = ['a', 'b', 'c', 'd']
        self.bool_cols = ['h']
        self.float_cols = ['e', 'f']

        self.con = MockConnection()
