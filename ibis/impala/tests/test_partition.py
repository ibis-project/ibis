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

import pytest

from pandas.util.testing import assert_frame_equal
import pandas as pd

from ibis.compat import unittest
from ibis.impala.compat import ImpylaError
from ibis.impala.tests.common import ImpalaE2E, IbisTestEnv
from ibis.tests.util import assert_equal
import ibis
import ibis.util as util


class TestPartitioning(ImpalaE2E, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestPartitioning, cls).setUpClass()
        df = pd.DataFrame({'year': [2009, 2009, 2009, 2010, 2010, 2010],
                           'month': [1, 2, 3, 1, 2, 3],
                           'value': [1, 2, 3, 4, 5, 6]})
        df = pd.concat([df] * 10, ignore_index=True)
        df['id'] = df.index.values

        cls.df = df
        cls.db = cls.con.database(cls.tmp_db)
        cls.pd_name = util.guid()
        cls.db.create_table(cls.pd_name, df, path=cls._create_777_tmp_dir())

    @pytest.mark.superuser
    def test_create_table_with_partition_column(self):
        schema = ibis.schema([('year', 'int32'),
                              ('month', 'int8'),
                              ('day', 'int8'),
                              ('value', 'double')])

        name = util.guid()
        self.con.create_table(name, schema=schema, partition=['year', 'month'],
                              path=self._create_777_tmp_dir())
        self.temp_tables.append(name)

        # the partition column get put at the end of the table
        ex_schema = ibis.schema([('day', 'int8'),
                                 ('value', 'double'),
                                 ('year', 'int32'),
                                 ('month', 'int8')])
        table_schema = self.con.get_schema(name)
        assert_equal(table_schema, ex_schema)

        partition_schema = self.con.table(name).partition_schema()

        expected = ibis.schema([('year', 'int32'),
                                ('month', 'int8')])
        assert_equal(partition_schema, expected)

    @pytest.mark.superuser
    def test_create_partitioned_separate_schema(self):
        schema = ibis.schema([('day', 'int8'),
                              ('value', 'double')])
        part_schema = ibis.schema([('year', 'int32'),
                                   ('month', 'int8')])

        name = util.guid()
        self.con.create_table(name, schema=schema, partition=part_schema,
                              path=self._create_777_tmp_dir())
        self.temp_tables.append(name)

        # the partition column get put at the end of the table
        ex_schema = ibis.schema([('day', 'int8'),
                                 ('value', 'double'),
                                 ('year', 'int32'),
                                 ('month', 'int8')])
        table_schema = self.con.get_schema(name)
        assert_equal(table_schema, ex_schema)

        partition_schema = self.con.table(name).partition_schema()
        assert_equal(partition_schema, part_schema)

    @pytest.mark.superuser
    def test_unpartitioned_table_get_schema(self):
        tname = 'functional_alltypes'
        with self.assertRaises(ImpylaError):
            self.con.table(tname).partition_schema()

    @pytest.mark.superuser
    def test_insert_select_partitioned_table(self):
        df = self.df

        unpart_t = self.db.table(self.pd_name)

        part_name = util.guid()

        part_keys = ['year', 'month']
        self.db.create_table(part_name,
                             schema=unpart_t.schema(),
                             partition=part_keys,
                             path=self._create_777_tmp_dir())

        part_t = self.db.table(part_name)
        unique_keys = df[part_keys].drop_duplicates()

        for i, (year, month) in enumerate(unique_keys.itertuples(index=False)):
            select_stmt = unpart_t[(unpart_t.year == year) &
                                   (unpart_t.month == month)]

            # test both styles of insert
            if i:
                part = {'year': year, 'month': month}
            else:
                part = [year, month]
            part_t.insert(select_stmt, partition=part)

        result = (part_t.execute()
                  .sort_index(by='id')
                  .reset_index(drop=True)
                  [df.columns])

        assert_frame_equal(result, df)

    @pytest.mark.superuser
    def test_insert_overwrite_partition(self):
        pass

    @pytest.mark.superuser
    def test_dynamic_partitioning(self):
        pass

    @pytest.mark.superuser
    def test_add_partition_with_location(self):
        pass

    @pytest.mark.superuser
    def test_set_partition_location(self):
        pass

    @pytest.mark.superuser
    def test_load_data_partition(self):
        pass

    @pytest.mark.superuser
    def test_repartition_automated(self):
        pass
