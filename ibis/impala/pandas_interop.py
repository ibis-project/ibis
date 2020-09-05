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

import csv
import os
import tempfile
from posixpath import join as pjoin

import ibis.common.exceptions as com
import ibis.expr.schema as sch
import ibis.util as util
from ibis.config import options


class DataFrameWriter:

    """
    Interface class for writing pandas objects to Impala tables

    Class takes ownership of any temporary data written to HDFS
    """

    def __init__(self, client, df, path=None):
        self.client = client
        self.hdfs = client.hdfs

        self.df = df

        self.temp_hdfs_dirs = []

    def write_temp_csv(self):
        temp_hdfs_dir = pjoin(
            options.impala.temp_hdfs_path, 'pandas_{}'.format(util.guid())
        )
        self.hdfs.mkdir(temp_hdfs_dir)

        # Keep track of the temporary HDFS file
        self.temp_hdfs_dirs.append(temp_hdfs_dir)

        # Write the file to HDFS
        hdfs_path = pjoin(temp_hdfs_dir, '0.csv')

        self.write_csv(hdfs_path)

        return temp_hdfs_dir

    def write_csv(self, path):
        # Use a temporary dir instead of a temporary file
        # to provide Windows support and avoid #2267
        # https://github.com/ibis-project/ibis/issues/2267
        with tempfile.TemporaryDirectory() as f:
            # Write the DataFrame to the temporary file path
            tmp_file_path = os.path.join(f, 'impala_temp_file.csv')
            if options.verbose:
                util.log(
                    'Writing DataFrame to temporary directory {}'.format(
                        tmp_file_path
                    )
                )

            self.df.to_csv(
                tmp_file_path,
                header=False,
                index=False,
                sep=',',
                quoting=csv.QUOTE_NONE,
                escapechar='\\',
                na_rep='#NULL',
            )

            if options.verbose:
                util.log('Writing CSV to: {0}'.format(path))

            self.hdfs.put(path, tmp_file_path)
        return path

    def get_schema(self):
        # define a temporary table using delimited data
        return sch.infer(self.df)

    def delimited_table(self, csv_dir, name=None, database=None):
        temp_delimited_name = 'ibis_tmp_pandas_{0}'.format(util.guid())
        schema = self.get_schema()

        return self.client.delimited_file(
            csv_dir,
            schema,
            name=temp_delimited_name,
            database=database,
            delimiter=',',
            na_rep='#NULL',
            escapechar='\\\\',
            external=True,
            persist=False,
        )

    def __del__(self):
        try:
            self.cleanup()
        except com.IbisError:
            pass

    def cleanup(self):
        for path in self.temp_hdfs_dirs:
            self.hdfs.rmdir(path)
        self.temp_hdfs_dirs = []
        self.csv_dir = None


def write_temp_dataframe(client, df):
    writer = DataFrameWriter(client, df)
    path = writer.write_temp_csv()
    return writer, writer.delimited_table(path)
