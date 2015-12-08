# Copyright 2015 Cloudera Inc.
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

import pytest

from ibis.compat import unittest
from ibis.impala.tests.common import IbisTestEnv, ImpalaE2E
from ibis.tests.util import assert_equal
import ibis.expr.datatypes as dt
import ibis

try:
    from ibis.impala import kudu_support as ksupport
    import kudu
    HAVE_KUDU_CLIENT = True
except ImportError:
    HAVE_KUDU_CLIENT = False


pytestmark = pytest.mark.skipif(not HAVE_KUDU_CLIENT,
                                reason='Kudu client not installed')


ENV = IbisTestEnv()


class TestKuduTools(unittest.TestCase):

    # Test schema conversion, DDL statements, etc.

    def test_kudu_schema_convert(self):
        spec = [
            # name, type, is_nullable, is_primary_key
            ('a', dt.Int8(False), 'int8', False, True),
            ('b', dt.Int16(False), 'int16', False, True),
            ('c', dt.Int32(False), 'int32', False, False),
            ('d', dt.Int64(True), 'int64', True, False),
            ('e', dt.String(True), 'string', True, False),
            ('f', dt.Boolean(False), 'bool', False, False),
            ('g', dt.Float(False), 'float', False, False),
            ('h', dt.Double(True), 'double', True, False),

            # TODO
            # ('i', 'binary', False, False),

            ('j', dt.Timestamp(True), 'timestamp', True, False)
        ]

        builder = kudu.schema_builder()
        primary_keys = []
        ibis_types = []
        for name, itype, type_, is_nullable, is_primary_key in spec:
            builder.add_column(name, type_, nullable=is_nullable)

            if is_primary_key:
                primary_keys.append(name)

            ibis_types.append((name, itype))

        builder.set_primary_keys(primary_keys)
        kschema = builder.build()

        ischema = ksupport.schema_kudu_to_ibis(kschema)
        expected = ibis.schema(ibis_types)

        assert_equal(ischema, expected)


class TestKuduE2E(ImpalaE2E, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.setup_e2e(cls)

    @classmethod
    def tearDownClass(cls):
        cls.teardown_e2e(cls)

    @pytest.mark.kudu
    def test_kudu_table(self):
        pass
