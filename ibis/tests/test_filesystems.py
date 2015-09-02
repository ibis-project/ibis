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

from six import BytesIO

from posixpath import join as pjoin
from os import path as osp
import os
import shutil

import pytest

from ibis.filesystems import HDFS
from ibis.compat import unittest
from ibis.impala.tests.common import IbisTestEnv
import ibis.compat as compat
import ibis.util as util
import ibis


ENV = IbisTestEnv()


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
        ls_contents = [(u'foo',
                        {u'type': u'DIRECTORY'}),
                       (u'bar.tmp',
                        {u'type': u'FILE'}),
                       (u'baz.copying',
                        {u'type': u'FILE'}),
                       (u'_SUCCESS',
                        {u'type': u'FILE'}),
                       (u'.peekaboo',
                        {u'type': u'FILE'}),
                       (u'0.parq',
                        {u'type': u'FILE'}),
                       (u'_FILE',
                        {u'type': u'DIRECTORY'})]

        self.con.set_ls(ls_contents)

        result = self.con._find_any_file('/path')
        assert result == '0.parq'


@pytest.mark.hdfs
class TestHDFSE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ENV = ENV
        cls.tmp_dir = pjoin(cls.ENV.tmp_dir, util.guid())
        if cls.ENV.auth_mechanism in ['GSSAPI', 'LDAP']:
            print("Warning: ignoring invalid Certificate Authority errors")
        cls.hdfs = ibis.hdfs_connect(host=cls.ENV.nn_host,
                                     port=cls.ENV.webhdfs_port,
                                     auth_mechanism=cls.ENV.auth_mechanism,
                                     verify=(cls.ENV.auth_mechanism
                                             not in ['GSSAPI', 'LDAP']))
        cls.hdfs.mkdir(cls.tmp_dir)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.hdfs.rmdir(cls.tmp_dir)
        except:
            pass

    def setUp(self):
        self.test_files = []
        self.test_directories = []

    def tearDown(self):
        self._delete_test_files()
        pass

    def _delete_test_files(self):
        for path in self.test_files:
            try:
                os.remove(path)
            except os.error:
                pass

        for path in self.test_directories:
            try:
                shutil.rmtree(path)
            except os.error:
                pass

    def _make_test_directory(self, files=5, filesize=1024, directory=None):
        if directory is None:
            directory = util.guid()
            os.mkdir(directory)
            self.test_directories.append(directory)

        for i in range(files):
            self._make_random_file(size=filesize, directory=directory)

        return directory

    def _make_random_file(self, size=1024, directory=None):
        path = util.guid()

        if directory:
            path = osp.join(directory, path)

        units = size / 32

        with open(path, 'wb') as f:
            for i in range(int(units)):
                f.write(guidbytes())

        self.test_files.append(path)
        return path

    def _make_random_hdfs_file(self, size=1024, directory=None):
        local_path = self._make_random_file(size=size)
        remote_path = pjoin(directory or self.tmp_dir, local_path)
        self.hdfs.put(remote_path, local_path)
        return remote_path

    def test_mkdir(self):
        path = pjoin(self.tmp_dir, 'mkdir-test')
        self.hdfs.mkdir(path)
        assert self.hdfs.exists(path)

    def test_chmod(self):
        new_permissions = '755'
        path = self._make_random_hdfs_file()
        self.hdfs.chmod(path, new_permissions)
        assert self.hdfs.status(path)['permission'] == new_permissions

    def test_chmod_directory(self):
        new_permissions = '755'
        path = pjoin(self.tmp_dir, util.guid())
        self.hdfs.mkdir(path)
        self.hdfs.chmod(path, new_permissions)
        assert self.hdfs.status(path)['permission'] == new_permissions

    def test_mv_to_existing_file(self):
        remote_file = self._make_random_hdfs_file()
        existing_remote_file_dest = self._make_random_hdfs_file()
        self.hdfs.mv(remote_file, existing_remote_file_dest)

    def test_mv_to_existing_file_no_overwrite(self):
        remote_file = self._make_random_hdfs_file()
        existing_remote_file_dest = self._make_random_hdfs_file()
        with self.assertRaises(Exception):
            self.hdfs.mv(remote_file, existing_remote_file_dest,
                         overwrite=False)

    def test_mv_to_directory(self):
        remote_file = self._make_random_hdfs_file()
        dest_dir = pjoin(self.tmp_dir, util.guid())
        self.hdfs.mkdir(dest_dir)
        self.hdfs.mv(remote_file, dest_dir)
        new_remote_file = pjoin(dest_dir, os.path.basename(remote_file))
        file_status = self.hdfs.status(new_remote_file)
        assert file_status['type'] == 'FILE'

    def test_put_get_delete_file(self):
        dirpath = pjoin(self.tmp_dir, 'write-delete-test')
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

    def test_put_get_directory(self):
        local_dir = util.guid()
        local_download_dir = util.guid()

        K = 5

        os.mkdir(local_dir)

        try:
            for i in range(K):
                self._make_random_file(directory=local_dir)

            remote_dir = pjoin(self.tmp_dir, local_dir)
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

    def test_put_file_into_directory(self):
        local_path = self._make_random_file()
        self.hdfs.put(self.tmp_dir, local_path)
        remote_file_path = pjoin(self.tmp_dir, local_path)
        file_status = self.hdfs.status(remote_file_path)
        assert file_status['type'] == 'FILE'

    def test_get_file_overwrite(self):
        local_path = self._make_random_file()
        local_path2 = self._make_random_file()

        remote_path = pjoin(self.tmp_dir, local_path)
        self.hdfs.put(remote_path, local_path)

        remote_path2 = pjoin(self.tmp_dir, local_path2)
        self.hdfs.put(remote_path2, local_path2)

        with self.assertRaises(Exception):
            self.hdfs.get(remote_path, '.')

        self.hdfs.get(remote_path, local_path2, overwrite=True)
        assert open(local_path2).read() == open(local_path).read()

    def test_put_buffer_like(self):
        data = b'peekaboo'

        buf = BytesIO()
        buf.write(data)
        buf.seek(0)

        remote_path = pjoin(self.tmp_dir, util.guid())
        self.hdfs.put(remote_path, buf)

        local_path = util.guid()
        self.test_files.append(local_path)

        self.hdfs.get(remote_path, local_path)
        assert open(local_path, 'rb').read() == data

    def test_get_logging(self):
        # TODO write a test for this
        pass

    def test_get_directory_nested_dirs(self):
        local_dir = util.guid()
        local_download_dir = util.guid()

        K = 5

        os.mkdir(local_dir)

        try:
            for i in range(K):
                self._make_random_file(directory=local_dir)

            nested_dir = osp.join(local_dir, 'nested-dir')
            shutil.copytree(local_dir, nested_dir)

            remote_dir = pjoin(self.tmp_dir, local_dir)
            self.hdfs.put(remote_dir, local_dir)

            # download directory and check contents
            self.hdfs.get(remote_dir, local_download_dir)

            _check_directories_equal(local_dir, local_download_dir)

            self._try_delete_directory(local_download_dir)

            self.hdfs.rmdir(remote_dir)
            assert not self.hdfs.exists(remote_dir)
        finally:
            shutil.rmtree(local_dir)

    def test_get_directory_overwrite_file(self):
        try:
            local_path1 = self._make_test_directory()
            local_path2 = self._make_random_file()
            remote_path = pjoin(self.tmp_dir, local_path1)
            self.hdfs.put(remote_path, local_path1)
            self.hdfs.get(remote_path, local_path2, overwrite=True)
            _check_directories_equal(local_path1, local_path2)
        finally:
            # Path changed from file to directory, must be cleaned manually.
            self._try_delete_directory(local_path2)

    def test_get_directory_overwrite_directory(self):
        local_path1 = self._make_test_directory()
        local_path2 = self._make_test_directory()
        remote_path = pjoin(self.tmp_dir, local_path2)
        self.hdfs.put(remote_path, local_path1)
        self.hdfs.get(remote_path, osp.dirname(local_path2), overwrite=True)
        _check_directories_equal(local_path1, local_path2)

    def test_get_directory_into_directory(self):
        local_path1 = self._make_test_directory()
        local_path2 = self._make_test_directory()
        remote_path = pjoin(self.tmp_dir, local_path1)
        self.hdfs.put(remote_path, local_path1)
        local_path3 = self.hdfs.get(remote_path, local_path2)
        _check_directories_equal(local_path3, local_path1)

    def _try_delete_directory(self, path):
        try:
            shutil.rmtree(path)
        except os.error:
            pass

    def test_ls(self):
        test_dir = pjoin(self.tmp_dir, 'ls-test')
        self.hdfs.mkdir(test_dir)
        for i in range(10):
            local_path = self._make_random_file()
            hdfs_path = pjoin(test_dir, local_path)
            self.hdfs.put(hdfs_path, local_path)
        assert len(self.hdfs.ls(test_dir)) == 10

    def test_size(self):
        test_dir = pjoin(self.tmp_dir, 'size-test')

        K = 2048
        path = self._make_random_file(size=K)
        hdfs_path = pjoin(test_dir, path)
        self.hdfs.put(hdfs_path, path)
        assert self.hdfs.size(hdfs_path) == K

        size_test_dir = self._sample_nested_directory()

        hdfs_path = pjoin(test_dir, size_test_dir)
        self.hdfs.put(hdfs_path, size_test_dir)

        assert self.hdfs.size(hdfs_path) == K * 7

    def test_put_get_tarfile(self):
        test_dir = pjoin(self.tmp_dir, 'tarfile-test')

        dirname = self._sample_nested_directory()

        import subprocess
        tf_name = '{0}.tar.gz'.format(dirname)
        cmd = 'tar zc {0} > {1}'.format(dirname, tf_name)

        retcode = subprocess.call(cmd, shell=True)
        if retcode:
            raise Exception((retcode, cmd))

        self.test_files.append(tf_name)

        randname = util.guid()
        hdfs_path = pjoin(test_dir, randname)
        self.hdfs.put_tarfile(hdfs_path, tf_name, compression='gzip')

        self.hdfs.get(hdfs_path, '.')
        self.test_directories.append(randname)
        _check_directories_equal(osp.join(randname, dirname), dirname)

    def _sample_nested_directory(self):
        K = 2048
        dirname = self._make_test_directory(files=2, filesize=K)
        nested_dir = osp.join(dirname, util.guid())
        os.mkdir(nested_dir)

        self._make_test_directory(files=5, filesize=K,
                                  directory=nested_dir)

        return dirname


@pytest.mark.hdfs
@pytest.mark.superuser
class TestSuperUserHDFSE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ENV = ENV
        cls.tmp_dir = pjoin(cls.ENV.tmp_dir, util.guid())
        if cls.ENV.auth_mechanism in ['GSSAPI', 'LDAP']:
            print("Warning: ignoring invalid Certificate Authority errors")
        # NOTE: specifying superuser as set in IbisTestEnv
        cls.hdfs = ibis.hdfs_connect(host=cls.ENV.nn_host,
                                     port=cls.ENV.webhdfs_port,
                                     auth_mechanism=cls.ENV.auth_mechanism,
                                     verify=(cls.ENV.auth_mechanism
                                             not in ['GSSAPI', 'LDAP']),
                                     user=cls.ENV.hdfs_superuser)
        cls.hdfs.mkdir(cls.tmp_dir)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.hdfs.rmdir(cls.tmp_dir)
        except:
            pass

    def setUp(self):
        self.test_files = []
        self.test_directories = []

    def tearDown(self):
        self._delete_test_files()
        pass

    def _delete_test_files(self):
        for path in self.test_files:
            try:
                os.remove(path)
            except os.error:
                pass

        for path in self.test_directories:
            try:
                shutil.rmtree(path)
            except os.error:
                pass

    def _make_random_file(self, size=1024, directory=None):
        path = util.guid()

        if directory:
            path = osp.join(directory, path)

        units = size / 32

        with open(path, 'wb') as f:
            for i in range(int(units)):
                f.write(guidbytes())

        self.test_files.append(path)
        return path

    def _make_random_hdfs_file(self, size=1024, directory=None):
        local_path = self._make_random_file(size=size)
        remote_path = pjoin(directory or self.tmp_dir, local_path)
        self.hdfs.put(remote_path, local_path)
        return remote_path

    def test_chown_owner(self):
        new_owner = 'randomowner'
        path = self._make_random_hdfs_file()
        self.hdfs.chown(path, new_owner)
        assert self.hdfs.status(path)['owner'] == new_owner

    def test_chown_group(self):
        new_group = 'randomgroup'
        path = self._make_random_hdfs_file()
        self.hdfs.chown(path, group=new_group)
        assert self.hdfs.status(path)['group'] == new_group

    def test_chown_group_directory(self):
        new_group = 'randomgroup'
        path = pjoin(self.tmp_dir, util.guid())
        self.hdfs.mkdir(path)
        self.hdfs.chown(path, group=new_group)
        assert self.hdfs.status(path)['group'] == new_group

    def test_chown_owner_directory(self):
        new_owner = 'randomowner'
        path = pjoin(self.tmp_dir, util.guid())
        self.hdfs.mkdir(path)
        self.hdfs.chown(path, new_owner)
        assert self.hdfs.status(path)['owner'] == new_owner


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
        rel_dir = osp.relpath(dirpath, path)
        if rel_dir == '.':
            rel_dir = ''
        for name in filenames:
            abspath = osp.join(dirpath, name)
            relpath = osp.join(rel_dir, name)
            paths[relpath] = abspath

    return paths


def guidbytes():
    if compat.PY3:
        return util.guid().encode('utf8')
    else:
        return util.guid()
