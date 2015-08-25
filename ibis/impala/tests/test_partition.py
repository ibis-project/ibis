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

import ibis

from ibis.compat import unittest
from ibis.impala.tests.common import ImpalaE2E
from ibis.tests.util import assert_equal

import ibis.util as util


class TestPartitioning(ImpalaE2E, unittest.TestCase):

    def test_create_table_with_partition_column(self):
        schema = ibis.schema([('year', 'int32'),
                              ('month', 'int8'),
                              ('day', 'int8'),
                              ('value', 'double')])

        name = util.guid()
        self.con.create_table(name, schema=schema, partition=['year', 'month'])
        self.temp_tables.append(name)

        # the partition column get put at the end of the table
        ex_schema = ibis.schema([('day', 'int8'),
                                 ('value', 'double'),
                                 ('year', 'int32'),
                                 ('month', 'int8')])
        table_schema = self.con.get_schema(name)
        assert_equal(table_schema, ex_schema)

        partition_schema = self.con.get_partition_schema(name)
        expected = ibis.schema([('year', 'int32'),
                                ('month', 'int8')])
        assert_equal(partition_schema, expected)
