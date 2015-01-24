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
            ('h', 'boolean')
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
        ]
    }

    def _get_table_schema(self, name):
        return ir.Schema.from_tuples(self._tables[name])
