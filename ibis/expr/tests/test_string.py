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

import unittest

import ibis.expr.base as api
import ibis.expr.base as ir
import ibis.expr.base as ops

from ibis.expr.tests.mocks import MockConnection


class TestStringOps(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_lower_upper(self):
        lresult = self.table.g.lower()
        uresult = self.table.g.upper()

        assert isinstance(lresult, ir.StringArray)
        assert isinstance(uresult, ir.StringArray)

        assert isinstance(lresult.op(), ops.Lowercase)
        assert isinstance(uresult.op(), ops.Uppercase)

        lit = api.literal('FoO')

        lresult = lit.lower()
        uresult = lit.upper()
        assert isinstance(lresult, ir.StringScalar)
        assert isinstance(uresult, ir.StringScalar)
