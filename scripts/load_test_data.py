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

from io import BytesIO
import os

import numpy as np
import pandas as pd
import pandas.util.testing as tm

from ibis.filesystems import HDFS

HDFS_HOST = 'localhost'
WEBHDFS_PORT = 5070

HDFS_USER = os.getlogin()

IBIS_TEST_FILE_DIRECTORY = '/ibis-test'


hdfs = HDFS(HDFS_HOST, WEBHDFS_PORT, params={'user.name': HDFS_USER})


def generate_test_data(clean_first=False):
    if clean_first:
        clean_all()

    make_base_directory()
    add_csv_test_files()


def make_base_directory():
    if not hdfs.path_exists(IBIS_TEST_FILE_DIRECTORY):
        hdfs.mkdir(IBIS_TEST_FILE_DIRECTORY, create_parent=True)


def add_csv_test_files(nfiles=10):
    N = 10

    path = 'csv-test'
    directory = '/'.join((IBIS_TEST_FILE_DIRECTORY, path))

    hdfs.rmdir(directory)

    df = pd.DataFrame({
        'foo': [tm.rands(10) for _ in xrange(N)],
        'bar': np.random.randn(N),
        'baz': np.random.randint(0, 100, size=N)
    }, columns=['foo', 'bar', 'baz'])

    buf = BytesIO()
    df.to_csv(buf, index=False, header=False)

    for i in xrange(nfiles):

        path = '/'.join((directory, '{}.csv'.format(i)))
        print('Writing {}'.format(path))

        buf.seek(0)
        hdfs.write(buf, path)


def clean_all():
    hdfs.rmdir(IBIS_TEST_FILE_DIRECTORY)


if __name__ == '__main__':
    generate_test_data()
