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

from ibis.compat import unittest
from ibis.tests.util import assert_equal
import ibis.expr.datatypes as dt
import ibis.sql.alchemy as alch
import ibis

import sqlalchemy as sa


class TestSQLAlchemy(unittest.TestCase):

    def setUp(self):
        self.meta = sa.MetaData()

    def test_sqla_schema_conversion(self):
        typespec = [
            # name, type, nullable
            ('one', sa.Integer, True, dt.int32),
            ('two', sa.SmallInteger, False, dt.int16),
            ('three', sa.REAL, True, dt.double),
            ('four', sa.Boolean, True, dt.boolean)
        ]

        sqla_types = []
        ibis_types = []
        for name, t, nullable, ibis_type in typespec:
            sqla_type = sa.Column(name, t, nullable=nullable)
            sqla_types.append(sqla_type)
            ibis_types.append((name, ibis_type(nullable)))

        table = sa.Table('tname', self.meta, *sqla_types)

        schema = alch.schema_from_table(table)
        expected = ibis.schema(ibis_types)

        assert_equal(schema, expected)
