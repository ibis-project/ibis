# Copyright 2015 Cloudera Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import ibis

from ibis.compat import unittest
from ibis.tests.util import IbisTestEnv, assert_equal

import ibis.common as com
import ibis.config as config
import ibis.expr.api as api
import ibis.expr.operations as _ops
import ibis.expr.types as ir
import ibis.util as util
import ibis.sql.udf as udf
import ibis.sql.exprs as _exprs

class UDFTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_sql_generation(self):
        op = udf.scalar_function(['string'], 'string', name='Tester')
        udf.add_impala_operation(op, 'identity', 'udf_testing')

        def _identity_test(value):
            return op(value).to_expr()
        result = _identity_test('hello world')
        
        assert result == "SELECT udf_testing.identity('hello world')"

    def test_sql_generation_from_infoclass(self):
        udf_info = udf.UDFInfo('test.so', 'info_test', ['string'], 'string')
        op = udf_info.to_operation()
        udf.add_impala_operation(op, 'info_test', 'udf_testing')

        def _infoclass_test(value):
            return op(value).to_expr()
        result = _infoclass_test('UDFInfo')

        assert result == "SELECT udf_testing.info_test('UDFInfo')"
