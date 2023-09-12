from __future__ import annotations

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
import contextlib
import csv
import os
import tempfile
from posixpath import join as pjoin

import ibis.common.exceptions as com
import ibis.expr.schema as sch
from ibis import util
from ibis.config import options


class DataFrameWriter:
    """Interface class for writing pandas objects to Impala tables.

    Notes
    -----
    Class takes ownership of any temporary data written to HDFS
    """

    def __init__(self, client, df):
        self.client = client
        self.df = df
        self.temp_hdfs_dirs = set()

    def write_temp_csv(self):
        temp_hdfs_dir = pjoin(options.impala.temp_hdfs_path, f"pandas_{util.guid()}")
        self.client.hdfs.mkdir(temp_hdfs_dir)

        # Keep track of the temporary HDFS file
        self.temp_hdfs_dirs.add(temp_hdfs_dir)

        # Write the file to HDFS
        hdfs_path = pjoin(temp_hdfs_dir, "0.csv")

        self.write_csv(hdfs_path)

        return temp_hdfs_dir

    def write_csv(self, path):
        # Use a temporary dir instead of a temporary file
        # to provide Windows support and avoid #2267
        # https://github.com/ibis-project/ibis/issues/2267
        with tempfile.TemporaryDirectory() as f:
            # Write the DataFrame to the temporary file path
            tmp_file_path = os.path.join(f, "impala_temp_file.csv")
            if options.verbose:
                util.log(f"Writing DataFrame to temporary directory {tmp_file_path}")

            self.df.to_csv(
                tmp_file_path,
                header=False,
                index=False,
                sep=",",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
                na_rep="#NULL",
            )

            if options.verbose:
                util.log(f"Writing CSV to: {path}")

            self.client.hdfs.put(tmp_file_path, path)
        return path

    def get_schema(self):
        # define a temporary table using delimited data
        return sch.infer(self.df)

    def delimited_table(self, csv_dir, database=None):
        return self.client.delimited_file(
            csv_dir,
            self.get_schema(),
            name=f"ibis_tmp_pandas_{util.guid()}",
            database=database,
            delimiter=",",
            na_rep="#NULL",
            escapechar="\\\\",
            external=True,
            persist=False,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with contextlib.suppress(com.IbisError):
            self.cleanup()
        return False

    def cleanup(self):
        while self.temp_hdfs_dirs:
            self.client.hdfs.rm(self.temp_hdfs_dirs.pop(), recursive=True)
