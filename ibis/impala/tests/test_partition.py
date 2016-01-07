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

from posixpath import join as pjoin
import pytest

from pandas.util.testing import assert_frame_equal
import pandas as pd

from ibis.compat import unittest
from ibis.impala.compat import ImpylaError
from ibis.impala.tests.common import ImpalaE2E, ENV
from ibis.tests.util import assert_equal
import ibis
import ibis.util as util


def _tmp_name():
    return 'tmp_partition_{0}'.format(util.guid())


class TestPartitioning(ImpalaE2E, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ImpalaE2E.setup_e2e(cls, ENV)

        df = pd.DataFrame({'year': [2009, 2009, 2009, 2010, 2010, 2010],
                           'month': ['1', '2', '3', '1', '2', '3'],
                           'value': [1, 2, 3, 4, 5, 6]})
        df = pd.concat([df] * 10, ignore_index=True)
        df['id'] = df.index.values

        cls.df = df
        cls.db = cls.con.database(cls.tmp_db)
        cls.pd_name = _tmp_name()
        cls.db.create_table(cls.pd_name, df)

    def test_is_partitioned(self):
        schema = ibis.schema([('foo', 'string'),
                              ('year', 'int32'),
                              ('month', 'string')])
        name = _tmp_name()
        self.db.create_table(name, schema=schema,
                             partition=['year', 'month'])
        assert self.db.table(name).is_partitioned

    def test_create_table_with_partition_column(self):
        schema = ibis.schema([('year', 'int32'),
                              ('month', 'string'),
                              ('day', 'int8'),
                              ('value', 'double')])

        name = _tmp_name()
        self.con.create_table(name, schema=schema,
                              database=self.tmp_db,
                              partition=['year', 'month'])
        self.temp_tables.append(name)

        # the partition column get put at the end of the table
        ex_schema = ibis.schema([('day', 'int8'),
                                 ('value', 'double'),
                                 ('year', 'int32'),
                                 ('month', 'string')])
        table_schema = self.con.get_schema(name, database=self.tmp_db)
        assert_equal(table_schema, ex_schema)

        partition_schema = self.db.table(name).partition_schema()

        expected = ibis.schema([('year', 'int32'),
                                ('month', 'string')])
        assert_equal(partition_schema, expected)

    def test_create_partitioned_separate_schema(self):
        schema = ibis.schema([('day', 'int8'),
                              ('value', 'double')])
        part_schema = ibis.schema([('year', 'int32'),
                                   ('month', 'string')])

        name = _tmp_name()
        self.con.create_table(name, schema=schema, partition=part_schema)
        self.temp_tables.append(name)

        # the partition column get put at the end of the table
        ex_schema = ibis.schema([('day', 'int8'),
                                 ('value', 'double'),
                                 ('year', 'int32'),
                                 ('month', 'string')])
        table_schema = self.con.get_schema(name)
        assert_equal(table_schema, ex_schema)

        partition_schema = self.con.table(name).partition_schema()
        assert_equal(partition_schema, part_schema)

    def test_unpartitioned_table_get_schema(self):
        tname = 'functional_alltypes'
        with self.assertRaises(ImpylaError):
            self.con.table(tname).partition_schema()

    def test_insert_select_partitioned_table(self):
        pytest.skip('IMPALA-2750')
        df = self.df

        unpart_t = self.db.table(self.pd_name)
        part_keys = ['year', 'month']
        part_t = self._create_partitioned_table(unpart_t.schema(),
                                                part_keys)
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

        self._verify_partitioned_table(part_t, df, unique_keys)

    def test_insert_overwrite_partition(self):
        pass

    def test_dynamic_partitioning(self):
        pass

    def test_add_drop_partition(self):
        pytest.skip('HIVE-12613')
        schema = ibis.schema([('foo', 'string'),
                              ('year', 'int32'),
                              ('month', 'int16')])
        name = _tmp_name()
        self.db.create_table(name, schema=schema,
                             partition=['year', 'month'])

        table = self.db.table(name)

        part = {'year': 2007, 'month': 4}

        path = '/tmp/tmp-{0}'.format(util.guid())
        table.add_partition(part, location=path)

        assert len(table.partitions()) == 2

        table.drop_partition(part)

        assert len(table.partitions()) == 1

    def test_set_partition_location(self):
        pass

    def test_load_data_partition(self):
        df = self.df

        unpart_t = self.db.table(self.pd_name)
        part_keys = ['year', 'month']
        part_t = self._create_partitioned_table(unpart_t.schema(),
                                                part_keys)

        # trim the runtime of this test
        df = df[df.month == '1'].reset_index(drop=True)

        unique_keys = df[part_keys].drop_duplicates()

        hdfs_dir = pjoin(self.tmp_dir, 'load-data-partition')

        df2 = df.drop(['year', 'month'], axis='columns')

        csv_props = {
            'serialization.format': ',',
            'field.delim': ','
        }

        for i, (year, month) in enumerate(unique_keys.itertuples(index=False)):
            chunk = df2[(df.year == year) & (df.month == month)]
            chunk_path = pjoin(hdfs_dir, '{0}.csv'.format(i))

            self.con.write_dataframe(chunk, chunk_path)

            # test both styles of insert
            if i:
                part = {'year': year, 'month': month}
            else:
                part = [year, month]

            part_t.add_partition(part)
            part_t.alter_partition(part, format='text',
                                   serde_properties=csv_props)
            part_t.load_data(chunk_path, partition=part)

        self.hdfs.rmdir(hdfs_dir)
        self._verify_partitioned_table(part_t, df, unique_keys)

    def _verify_partitioned_table(self, part_t, df, unique_keys):
        result = (part_t.execute()
                  .sort_index(by='id')
                  .reset_index(drop=True)
                  [df.columns])

        assert_frame_equal(result, df)

        parts = part_t.partitions()

        # allow for the total line
        assert len(parts) == (len(unique_keys) + 1)

    def _create_partitioned_table(self, schema, part_keys, location=None):
        part_name = _tmp_name()

        self.db.create_table(part_name,
                             schema=schema,
                             partition=part_keys)
        self.temp_tables.append(part_name)
        return self.db.table(part_name)

    def test_drop_partition(self):
        pass

    def test_repartition_automated(self):
        pass
