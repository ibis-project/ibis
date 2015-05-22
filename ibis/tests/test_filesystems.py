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

from ibis.filesystems import HDFS

import pytest
import unittest


class MockHDFS(HDFS):

    def __init__(self):
        self.ls_result = []

    def set_ls(self, results):
        self.ls_result = results

    def ls(self, *args, **kwargs):
        return self.ls_result


class TestHDFSRandom(unittest.TestCase):

    def setUp(self):
        self.con = MockHDFS()

    def test_find_any_file(self):
        ls_contents = [(u'/path/foo',
                        {u'type': u'DIRECTORY'}),
                       (u'/path/bar.tmp',
                        {u'type': u'FILE'}),
                       (u'/path/baz.copying',
                        {u'type': u'FILE'}),
                       (u'/path/_SUCCESS',
                        {u'type': u'FILE'}),
                       (u'/path/.peekaboo',
                        {u'type': u'FILE'}),
                       (u'/path/0.parq',
                        {u'type': u'FILE'}),
                       (u'/path/_FILE',
                        {u'type': u'DIRECTORY'})]

        self.con.set_ls(ls_contents)

        result = self.con.find_any_file('/path')
        assert result == '/path/0.parq'
