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

from ibis.compat import unittest
import ibis.sql.udf as udf
from ibis.sql.exprs import _operation_registry
from ibis.expr.operations import ValueOp


class UDFTest(unittest.TestCase):

    def test_sql_generation(self):
        op = udf.scalar_function(['string'], 'string', name='Tester')
        udf.add_impala_operation(op, 'identity', 'udf_testing')

        def _identity_test(value):
            return op(value).to_expr()
        result = _identity_test('hello world')
        assert result == "SELECT udf_testing.identity('hello world')"

    def test_sql_generation_from_infoclass(self):
        udf_info = udf.UDFCreator('test.so', ['string'], 'string', 'info_test')
        op = udf_info.to_operation()
        udf.add_impala_operation(op, 'info_test', 'udf_testing')
        assert op in _operation_registry

        def _infoclass_test(value):
            return op(value).to_expr()
        result = _infoclass_test('hello world')

        assert result == "SELECT udf_testing.info_test('hello world')"

    def test_udf_class_creation(self):
        op = udf.scalar_function(['string'], 'string', name='Tester')
        assert issubclass(op, ValueOp)
