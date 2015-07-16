#! /usr/bin/env python
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

# Cleans up the ibis-testing-data from Impala/HDFS and also the HDFS tmp data
# directory

from __future__ import print_function

from posixpath import join as pjoin
import os
import posixpath
import shutil
import sys
import tempfile
import subprocess

import ibis
from ibis.tests.util import IbisTestEnv


ENV = IbisTestEnv()


def make_connection():
    ic = ibis.impala_connect(host=ENV.impala_host, port=ENV.impala_port,
                             protocol=ENV.impala_protocol,
                             use_kerberos=ENV.use_kerberos)
    if ENV.use_kerberos:
        print("Warning: ignoring invalid Certificate Authority errors")
    hdfs = ibis.hdfs_connect(host=ENV.nn_host, port=ENV.webhdfs_port,
                             use_kerberos=ENV.use_kerberos,
                             verify=(not ENV.use_kerberos))
    return ibis.make_client(ic, hdfs_client=hdfs)


if __name__ == '__main__':
    if ENV.cleanup_test_data:
        con = make_connection()
        con.drop_database(ENV.test_data_db, force=True)
        con.hdfs.rmdir(ENV.test_data_dir)
        con.hdfs.rmdir(ENV.tmp_dir)
    else:
        print('IBIS_TEST_CLEANUP_TEST_DATA not set to True; refusing to clean')
