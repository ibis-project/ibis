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
import os

import pandas.core.common as pdcom
import pandas as pd

import ibis.common as com

from ibis.config import options
from ibis.util import log
import ibis.compat as compat
import ibis.expr.datatypes as itypes
import ibis.util as util


# ----------------------------------------------------------------------
# pandas integration


def pandas_col_to_ibis_type(col):
    import numpy as np
    dty = col.dtype

    # datetime types
    if pdcom.is_datetime64_dtype(dty):
        if pdcom.is_datetime64_ns_dtype(dty):
            return 'timestamp'
        else:
            raise com.IbisTypeError("Column {0} has dtype {1}, which is "
                                    "datetime64-like but does "
                                    "not use nanosecond units"
                                    .format(col.name, dty))
    if pdcom.is_timedelta64_dtype(dty):
        print("Warning: encoding a timedelta64 as an int64")
        return 'int64'

    if pdcom.is_categorical_dtype(dty):
        return itypes.Category(len(col.cat.categories))

    if pdcom.is_bool_dtype(dty):
        return 'boolean'

    # simple numerical types
    if issubclass(dty.type, np.int8):
        return 'int8'
    if issubclass(dty.type, np.int16):
        return 'int16'
    if issubclass(dty.type, np.int32):
        return 'int32'
    if issubclass(dty.type, np.int64):
        return 'int64'
    if issubclass(dty.type, np.float32):
        return 'float'
    if issubclass(dty.type, np.float64):
        return 'double'
    if issubclass(dty.type, np.uint8):
        return 'int16'
    if issubclass(dty.type, np.uint16):
        return 'int32'
    if issubclass(dty.type, np.uint32):
        return 'int64'
    if issubclass(dty.type, np.uint64):
        raise com.IbisTypeError("Column {0} is an unsigned int64"
                                .format(col.name))

    if pdcom.is_object_dtype(dty):
        return _infer_object_dtype(col)

    raise com.IbisTypeError("Column {0} is dtype {1}".format(col.name, dty))


def _infer_object_dtype(arr):
    # TODO: accelerate with Cython/C

    BOOLEAN, STRING = 0, 1
    state = BOOLEAN

    avalues = arr.values if isinstance(arr, pd.Series) else arr
    nulls = pd.isnull(avalues)

    if nulls.any():
        for i in compat.range(len(avalues)):
            if state == BOOLEAN:
                if not nulls[i] and not pdcom.is_bool(avalues[i]):
                    state = STRING
            elif state == STRING:
                break
        if state == BOOLEAN:
            return 'boolean'
        elif state == STRING:
            return 'string'
    else:
        return pd.lib.infer_dtype(avalues)


class DataFrameWriter(object):

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
        temp_hdfs_dir = pjoin(options.impala.temp_hdfs_path,
                              'pandas_{0}'.format(util.guid()))
        self.hdfs.mkdir(temp_hdfs_dir)

        # Keep track of the temporary HDFS file
        self.temp_hdfs_dirs.append(temp_hdfs_dir)

        # Write the file to HDFS
        hdfs_path = pjoin(temp_hdfs_dir, '0.csv')

        self.write_csv(hdfs_path)

        return temp_hdfs_dir

    def write_csv(self, path):
        import csv

        tmp_path = 'tmp_{0}.csv'.format(util.guid())
        f = open(tmp_path, 'w+')

        try:
            # Write the DataFrame to the temporary file path
            if options.verbose:
                log('Writing DataFrame to temporary file')

            self.df.to_csv(f, header=False, index=False,
                           sep=',',
                           quoting=csv.QUOTE_NONE,
                           escapechar='\\',
                           na_rep='#NULL')
            f.seek(0)

            if options.verbose:
                log('Writing CSV to: {0}'.format(path))

            self.hdfs.put(path, f)
        finally:
            f.close()
            try:
                os.remove(tmp_path)
            except os.error:
                pass

        return path

    def get_schema(self):
        # define a temporary table using delimited data
        return pandas_to_ibis_schema(self.df)

    def delimited_table(self, csv_dir, name=None, database=None):
        temp_delimited_name = 'ibis_tmp_pandas_{0}'.format(util.guid())
        schema = self.get_schema()

        return self.client.delimited_file(csv_dir, schema,
                                          name=temp_delimited_name,
                                          database=database,
                                          delimiter=',',
                                          na_rep='#NULL',
                                          escapechar='\\\\',
                                          external=True,
                                          persist=False)

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


def pandas_to_ibis_schema(frame):
    from ibis.expr.api import schema
    # no analog for decimal in pandas
    pairs = []
    for col_name in frame:
        ibis_type = pandas_col_to_ibis_type(frame[col_name])
        pairs.append((col_name, ibis_type))
    return schema(pairs)


def write_temp_dataframe(client, df):
    writer = DataFrameWriter(client, df)
    path = writer.write_temp_csv()
    return writer, writer.delimited_table(path)
