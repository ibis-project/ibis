# Copyright 2015 Cloudera Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from posixpath import join as pjoin
import pytest

from ibis.compat import unittest
from ibis.impala.tests.common import ImpalaE2E

from ibis.impala import madlib

import ibis.util as util


class TestMADLib(ImpalaE2E, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestMADLib, cls).setUpClass()
        cls.db = '__ibis_madlib_{0}'.format(util.guid()[:4])

        cls.con.create_database(cls.db)

    @classmethod
    def tearDownClass(cls):
        super(TestMADLib, cls).tearDownClass()

        try:
            cls.con.drop_database(cls.db, force=True)
        except:
            pass

    def setUp(self):
        super(TestMADLib, self).setUp()
        self.madlib_so = pjoin(self.test_data_dir, 'udf/madlib.so')

        self.api = madlib.MADLibAPI(self.madlib_so, self.db)

    @pytest.mark.madlib
    def test_create_functions(self):
        self.api.create_functions(self.con)

        for name in self.api._udfs:
            func = getattr(self.api, name)
            assert self.con.exists_udf(func.name, database=self.db)

        for name in self.api._udas:
            func = getattr(self.api, name)
            assert self.con.exists_uda(func.name, database=self.db)
