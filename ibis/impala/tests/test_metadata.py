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

import pandas as pd

from numpy import nan

from ibis.impala.metadata import parse_metadata


def _glue_lists_spacer(spacer, lists):
    result = list(lists[0])
    for lst in lists[1:]:
        result.append(spacer)
        result.extend(lst)
    return result


class TestMetadataParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spacer = ('', nan, nan)

        cls.schema = [
            ('# col_name', 'data_type', 'comment'),
            cls.spacer,
            ('foo', 'int', nan),
            ('bar', 'tinyint', nan),
            ('baz', 'bigint', nan)
        ]

        cls.partitions = [
            ('# Partition Information', nan, nan),
            ('# col_name', 'data_type', 'comment'),
            cls.spacer,
            ('qux', 'bigint', nan)
        ]

        cls.info = [
            ('# Detailed Table Information', nan, nan),
            ('Database:', 'tpcds', nan),
            ('Owner:', 'wesm', nan),
            ('CreateTime:', 'Sun Nov 08 01:09:42 PST 2015', nan),
            ('LastAccessTime:', 'UNKNOWN', nan),
            ('Protect Mode:', 'None', nan),
            ('Retention:', '0', nan),
            ('Location:', ('hdfs://host-name:20500/my.db'
                           '/dbname.table_name'), nan),
            ('Table Type:', 'EXTERNAL_TABLE', nan),
            ('Table Parameters:', nan, nan),
            ('', 'EXTERNAL', 'TRUE'),
            ('', 'STATS_GENERATED_VIA_STATS_TASK', 'true'),
            ('', 'numRows', '183592'),
            ('', 'transient_lastDdlTime', '1447340941'),
        ]

        cls.storage_info = [
            ('# Storage Information', nan, nan),
            ('SerDe Library:', ('org.apache.hadoop'
                                '.hive.serde2.lazy.LazySimpleSerDe'), nan),
            ('InputFormat:', 'org.apache.hadoop.mapred.TextInputFormat', nan),
            ('OutputFormat:',
             'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
             nan),
            ('Compressed:', 'No', nan),
            ('Num Buckets:', '0', nan),
            ('Bucket Columns:', '[]', nan),
            ('Sort Columns:', '[]', nan),
            ('Storage Desc Params:', nan, nan),
            ('', 'field.delim', '|'),
            ('', 'serialization.format', '|')
        ]

        cls.part_metadata = pd.DataFrame.from_records(
            _glue_lists_spacer(cls.spacer, [cls.schema, cls.partitions,
                                            cls.info, cls.storage_info]),
            columns=['name', 'type', 'comment'])

        cls.unpart_metadata = pd.DataFrame.from_records(
            _glue_lists_spacer(cls.spacer, [cls.schema, cls.info,
                                            cls.storage_info]),
            columns=['name', 'type', 'comment'])

        cls.parsed_part = parse_metadata(cls.part_metadata)
        cls.parsed_unpart = parse_metadata(cls.unpart_metadata)

    def test_table_params(self):
        params = self.parsed_part.info['Table Parameters']

        assert params['EXTERNAL'] is True
        assert params['STATS_GENERATED_VIA_STATS_TASK'] is True
        assert params['numRows'] == 183592
        assert (params['transient_lastDdlTime'] ==
                pd.Timestamp('2015-11-12 15:09:01'))

    def test_partitions(self):
        assert self.parsed_unpart.partitions is None
        assert self.parsed_part.partitions == [('qux', 'bigint')]

    def test_schema(self):
        assert self.parsed_part.schema == [
            ('foo', 'int'),
            ('bar', 'tinyint'),
            ('baz', 'bigint')
        ]

    def test_storage_info(self):
        storage = self.parsed_part.storage
        assert storage['Compressed'] is False
        assert storage['Num Buckets'] == 0

    def test_storage_params(self):
        params = self.parsed_part.storage['Desc Params']

        assert params['field.delim'] == '|'
        assert params['serialization.format'] == '|'
