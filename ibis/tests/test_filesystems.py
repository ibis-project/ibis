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
import shutil
import unittest

from hdfs import InsecureClient
import pytest

from ibis.filesystems import HDFS, WebHDFS
import ibis.util as util


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


class TestHDFSE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.host = os.environ.get('IBIS_TEST_HOST', 'localhost')
        cls.protocol = os.environ.get('IBIS_TEST_PROTOCOL', 'hiveserver2')
        cls.port = os.environ.get('IBIS_TEST_PORT', 21050)
        cls.hdfs_host = os.environ.get('IBIS_TEST_HDFS_HOST', 'localhost')
        # Impala dev environment uses port 5070 for HDFS web interface
        cls.webhdfs_port = os.environ.get('IBIS_TEST_WEBHDFS_PORT', 5070)
        url = 'http://{}:{}'.format(cls.hdfs_host, cls.webhdfs_port)

        cls.test_dir = '/{}'.format(util.guid())

        try:
            cls.hdfs_client = InsecureClient(url)
            cls.hdfs = WebHDFS(cls.hdfs_client)
            cls.hdfs.mkdir(cls.test_dir)
        except Exception as e:
            pytest.skip('Could not connect to HDFS: {}'.format(e.message))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.hdfs.rmdir(cls.test_dir)
        except:
            pass

    def setUp(self):
        self.test_files = []

    def tearDown(self):
        self._delete_test_files()

    def _delete_test_files(self):
        for path in self.test_files:
            try:
                os.remove(path)
            except os.error:
                pass

    def _make_random_file(self, units=100, directory=None):
        path = util.guid()

        if directory:
            path = os.path.join(directory, path)

        with open(path, 'wb') as f:
            for i in xrange(units):
                f.write(util.guid())

        self.test_files.append(path)
        return path

    def test_mkdir(self):
        path = pjoin(self.test_dir, 'mkdir-test')
        self.hdfs.mkdir(path)
        assert self.hdfs.exists(path)

    def test_write_get_delete_file(self):
        dirpath = pjoin(self.test_dir, 'write-delete-test')
        self.hdfs.mkdir(dirpath)

        lpath = self._make_random_file()
        fpath = pjoin(dirpath, lpath)

        self.hdfs.put(fpath, lpath)
        assert self.hdfs.exists(fpath)

        try:
            dpath = util.guid()
            self.hdfs.get(fpath, dpath)
            assert _contents_equal(dpath, lpath)
            os.remove(dpath)
        finally:
            self.hdfs.rm(fpath)
            assert not self.hdfs.exists(fpath)

    def test_overwrite_file(self):
        pass

    def test_write_get_directory(self):
        local_dir = util.guid()
        local_download_dir = util.guid()

        K = 5

        os.mkdir(local_dir)

        try:
            for i in xrange(K):
                self._make_random_file(directory=local_dir)

            remote_dir = pjoin(self.test_dir, local_dir)
            self.hdfs.put(remote_dir, local_dir)

            assert self.hdfs.exists(remote_dir)
            assert len(self.hdfs.ls(remote_dir)) == K

            # download directory and check contents
            self.hdfs.get(remote_dir, local_download_dir)

            _check_directories_equal(local_dir, local_download_dir)

            self._try_delete_directory(local_download_dir)

            self.hdfs.rmdir(remote_dir)
            assert not self.hdfs.exists(remote_dir)
        finally:
            shutil.rmtree(local_dir)

    def test_get_directory_nested_dirs(self):
        local_dir = util.guid()
        local_download_dir = util.guid()

        K = 5

        os.mkdir(local_dir)

        try:
            for i in xrange(K):
                self._make_random_file(directory=local_dir)

            nested_dir = os.path.join(local_dir, 'nested-dir')
            shutil.copytree(local_dir, nested_dir)

            remote_dir = pjoin(self.test_dir, local_dir)
            self.hdfs.put(remote_dir, local_dir)

            # download directory and check contents
            self.hdfs.get(remote_dir, local_download_dir)

            _check_directories_equal(local_dir, local_download_dir)

            self._try_delete_directory(local_download_dir)

            self.hdfs.rmdir(remote_dir)
            assert not self.hdfs.exists(remote_dir)
        finally:
            shutil.rmtree(local_dir)

    def _try_delete_directory(self, path):
        try:
            shutil.rmtree(path)
        except os.error:
            pass

    def test_ls(self):
        test_dir = pjoin(self.test_dir, 'ls-test')
        self.hdfs.mkdir(test_dir)
        for i in xrange(10):
            local_path = self._make_random_file()
            hdfs_path = pjoin(test_dir, local_path)
            self.hdfs.put(hdfs_path, local_path)

        assert len(self.hdfs.ls(test_dir)) == 10


def _check_directories_equal(left, right):
    left_files = _get_all_files(left)
    right_files = _get_all_files(right)

    assert set(left_files.keys()) == set(right_files.keys())

    for relpath, labspath in left_files.items():
        rabspath = right_files[relpath]
        assert _contents_equal(rabspath, labspath)


def _contents_equal(left, right):
    with open(left) as lf:
        with open(right) as rf:
            return lf.read() == rf.read()


def _get_all_files(path):
    paths = {}
    for dirpath, _, filenames in os.walk(path):
        rel_dir = os.path.relpath(dirpath, path)
        if rel_dir == '.':
            rel_dir = ''
        for name in filenames:
            abspath = os.path.join(dirpath, name)
            relpath = os.path.join(rel_dir, name)
            paths[relpath] = abspath

    return paths
