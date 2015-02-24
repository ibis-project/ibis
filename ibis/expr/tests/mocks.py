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

from ibis.connection import SQLConnection
import ibis.expr.base as ir


class MockConnection(SQLConnection):

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
            ('l_quantity', 'decimal'),
            ('l_extendedprice', 'decimal'),
            ('l_discount', 'decimal'),
            ('l_tax', 'decimal'),
            ('l_returnflag', 'string'),
            ('l_linestatus', 'string'),
            ('l_shipdate', 'string'),
            ('l_commitdate', 'string'),
            ('l_receiptdate', 'string'),
            ('l_shipinstruct', 'string'),
            ('l_shipmode', 'string'),
            ('l_comment', 'string')
        ]
    }

    def _get_table_schema(self, name):
        return ir.Schema.from_tuples(self._tables[name])
