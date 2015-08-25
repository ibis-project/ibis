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

from ibis.sql.compiler import to_sql
from ibis.compat import unittest


class TestImpalaSQL(unittest.TestCase):

    def test_relabel_projection(self):
        # GH #551
        types = ['int32', 'string', 'double']
        table = ibis.table(zip(['foo', 'bar', 'baz'], types), 'table')
        relabeled = table.relabel({'foo': 'one', 'baz': 'three'})

        result = to_sql(relabeled)
        expected = """\
SELECT `foo` AS `one`, `bar`, `baz` AS `three`
FROM `table`"""
        assert result == expected
