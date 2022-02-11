import filecmp
import os
import shutil
import subprocess
from io import BytesIO
from os import path as osp
from pathlib import Path
from posixpath import join as pjoin

import pytest

import ibis
import ibis.util as util

pytest.importorskip("hdfs")
pytest.importorskip("impala")

from hdfs.util import HdfsError  # noqa: E402

from ibis.backends.impala.hdfs import HDFS  # noqa: E402
from ibis.backends.impala.tests.conftest import IbisTestEnv  # noqa: E402

pytestmark = pytest.mark.hdfs


@pytest.fixture
def mockhdfs(mocker):
    mocker.patch.multiple(
        HDFS,
        ls=lambda *args, **kwargs: [
            ('foo', {'type': 'DIRECTORY'}),
            ('bar.tmp', {'type': 'FILE'}),
            ('baz.copying', {'type': 'FILE'}),
            ('_SUCCESS', {'type': 'FILE'}),
            ('.peekaboo', {'type': 'FILE'}),
            ('0.parq', {'type': 'FILE'}),
            ('_FILE', {'type': 'DIRECTORY'}),
        ],
    )
    return HDFS()


def test_find_any_file(mockhdfs):
    result = mockhdfs._find_any_file('/path')
    assert result == '0.parq'


@pytest.fixture
def env():
    return IbisTestEnv()


@pytest.fixture
def tmp_dir(env):
    path = pjoin(env.tmp_dir, util.guid())
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def hdfs(env, tmp_dir):
    hdfs = ibis.impala.hdfs_connect(
        host=env.nn_host,
        port=env.webhdfs_port,
        auth_mechanism=env.auth_mechanism,
        verify=env.auth_mechanism not in ['GSSAPI', 'LDAP'],
        user=env.webhdfs_user,
    )
    hdfs.mkdir(tmp_dir)
    try:
        yield hdfs
    finally:
        hdfs.rmdir(tmp_dir)


def make_test_directory(files=5, filesize=1024, directory=None):
    if directory is None:
        directory = util.guid()
        os.mkdir(directory)

    for _ in range(files):
        make_random_file(size=filesize, directory=directory)

    return directory


def make_random_file(size=1024, directory=None):
    path = util.guid()

    if directory:
        path = osp.join(directory, path)

    Path(path).write_bytes(os.urandom(size))

    return path


@pytest.fixture
def random_file():
    path = make_random_file()
    try:
        yield path
    finally:
        os.remove(path)


@pytest.fixture
def random_file_dest():
    path = make_random_file()
    try:
        yield path
    finally:
        os.remove(path)


@pytest.fixture
def random_hdfs_file(hdfs, random_file, tmp_dir):
    local_path = random_file
    remote_path = pjoin(tmp_dir, local_path)
    hdfs.put(remote_path, local_path)
    try:
        yield remote_path
    finally:
        hdfs.rm(remote_path)


@pytest.fixture
def random_hdfs_file_dest(hdfs, random_file_dest, tmp_dir):
    local_path = random_file_dest
    remote_path = pjoin(tmp_dir, local_path)
    hdfs.put(remote_path, local_path)
    try:
        yield remote_path
    finally:
        hdfs.rm(remote_path)


def test_mkdir(hdfs, tmp_dir):
    path = pjoin(tmp_dir, 'mkdir-test')
    hdfs.mkdir(path)
    assert hdfs.exists(path)


def test_chmod(hdfs, tmp_dir, random_hdfs_file):
    new_permissions = '755'
    path = random_hdfs_file
    hdfs.chmod(path, new_permissions)
    assert hdfs.status(path)['permission'] == new_permissions


def test_chmod_directory(hdfs, tmp_dir):
    new_permissions = '755'
    path = pjoin(tmp_dir, util.guid())
    hdfs.mkdir(path)
    hdfs.chmod(path, new_permissions)
    assert hdfs.status(path)['permission'] == new_permissions


def test_mv_to_existing_file(
    hdfs,
    tmp_dir,
    random_hdfs_file,
    random_hdfs_file_dest,
):
    remote_file = random_hdfs_file
    existing_remote_file_dest = random_hdfs_file_dest
    hdfs.mv(remote_file, existing_remote_file_dest)


def test_mv_to_existing_file_no_overwrite(
    hdfs,
    random_hdfs_file,
    random_hdfs_file_dest,
):
    remote_file = random_hdfs_file
    existing_remote_file_dest = random_hdfs_file_dest
    with pytest.raises(HdfsError):
        hdfs.mv(remote_file, existing_remote_file_dest, overwrite=False)


def test_mv_to_directory(hdfs, tmp_dir, random_hdfs_file):
    remote_file = random_hdfs_file
    dest_dir = pjoin(tmp_dir, util.guid())
    hdfs.mkdir(dest_dir)
    hdfs.mv(remote_file, dest_dir)
    new_remote_file = pjoin(dest_dir, os.path.basename(remote_file))
    file_status = hdfs.status(new_remote_file)
    assert file_status['type'] == 'FILE'


def test_put_get_delete_file(hdfs, tmp_dir, random_file):
    dirpath = pjoin(tmp_dir, 'write-delete-test')
    hdfs.mkdir(dirpath)

    lpath = random_file
    fpath = pjoin(dirpath, lpath)

    hdfs.put(fpath, lpath)
    assert hdfs.exists(fpath)

    try:
        dpath = util.guid()
        hdfs.get(fpath, dpath)
        assert filecmp.cmp(dpath, lpath, shallow=False)
        os.remove(dpath)
    finally:
        hdfs.rm(fpath)
        assert not hdfs.exists(fpath)


def test_put_get_directory(hdfs, tmp_dir):
    local_dir = util.guid()
    local_download_dir = util.guid()

    K = 5

    os.mkdir(local_dir)

    try:
        for _ in range(K):
            make_random_file(directory=local_dir)

        remote_dir = pjoin(tmp_dir, local_dir)
        hdfs.put(remote_dir, local_dir)

        assert hdfs.exists(remote_dir)
        assert len(hdfs.ls(remote_dir)) == K

        # download directory and check contents
        hdfs.get(remote_dir, local_download_dir)

        _check_directories_equal(local_dir, local_download_dir)

        shutil.rmtree(local_download_dir, ignore_errors=True)

        hdfs.rmdir(remote_dir)
        assert not hdfs.exists(remote_dir)
    finally:
        shutil.rmtree(local_dir)


def test_put_file_into_directory(hdfs, tmp_dir, random_file):
    local_path = random_file
    hdfs.put(tmp_dir, local_path)
    remote_file_path = pjoin(tmp_dir, local_path)
    file_status = hdfs.status(remote_file_path)
    assert file_status['type'] == 'FILE'


def test_get_file_overwrite(hdfs, tmp_dir, random_file, random_file_dest):
    local_path = random_file
    local_path2 = random_file_dest

    remote_path = pjoin(tmp_dir, local_path)
    hdfs.put(remote_path, local_path)

    remote_path2 = pjoin(tmp_dir, local_path2)
    hdfs.put(remote_path2, local_path2)

    with pytest.raises(HdfsError):
        hdfs.get(remote_path, '.')

    hdfs.get(remote_path, local_path2, overwrite=True)
    assert Path(local_path2).read_bytes() == Path(local_path).read_bytes()


def test_put_buffer_like(hdfs, tmp_dir):
    data = b'peekaboo'

    remote_path = pjoin(tmp_dir, util.guid())
    hdfs.put(remote_path, BytesIO(data))

    local_path = pjoin(tmp_dir, util.guid())

    hdfs.get(remote_path, local_path)
    assert Path(local_path).read_bytes() == data


def test_get_directory_nested_dirs(hdfs, tmp_dir):
    local_dir = util.guid()
    local_download_dir = util.guid()

    K = 5

    os.mkdir(local_dir)

    try:
        for _ in range(K):
            make_random_file(directory=local_dir)

        nested_dir = osp.join(local_dir, 'nested-dir')
        shutil.copytree(local_dir, nested_dir)

        remote_dir = pjoin(tmp_dir, local_dir)
        hdfs.put(remote_dir, local_dir)

        # download directory and check contents
        hdfs.get(remote_dir, local_download_dir)

        _check_directories_equal(local_dir, local_download_dir)

        shutil.rmtree(local_download_dir, ignore_errors=True)

        hdfs.rmdir(remote_dir)
        assert not hdfs.exists(remote_dir)
    finally:
        shutil.rmtree(local_dir)


def test_get_directory_overwrite_file(hdfs, tmp_dir):
    local_path1 = make_test_directory()
    try:
        local_path2 = make_random_file()
        try:
            remote_path = pjoin(tmp_dir, local_path1)
            hdfs.put(remote_path, local_path1)
            hdfs.get(remote_path, local_path2, overwrite=True)
            _check_directories_equal(local_path1, local_path2)
        finally:
            # Path changed from file to directory, must be cleaned manually.
            shutil.rmtree(local_path2, ignore_errors=True)
    finally:
        shutil.rmtree(local_path1, ignore_errors=True)


def test_get_directory_overwrite_directory(hdfs, tmp_dir):
    local_path1 = make_test_directory()
    try:
        local_path2 = make_test_directory()
        try:
            remote_path = pjoin(tmp_dir, local_path2)
            hdfs.put(remote_path, local_path1)
            hdfs.get(remote_path, osp.dirname(local_path2), overwrite=True)
            _check_directories_equal(local_path1, local_path2)
        finally:
            shutil.rmtree(local_path2, ignore_errors=True)
    finally:
        shutil.rmtree(local_path1, ignore_errors=True)


def test_get_directory_into_directory(hdfs, tmp_dir):
    local_path1 = make_test_directory()
    try:
        local_path2 = make_test_directory()
        try:
            remote_path = pjoin(tmp_dir, local_path1)
            hdfs.put(remote_path, local_path1)
            local_path3 = hdfs.get(remote_path, local_path2)
            _check_directories_equal(local_path3, local_path1)
        finally:
            shutil.rmtree(local_path2, ignore_errors=True)
    finally:
        shutil.rmtree(local_path1, ignore_errors=True)


def test_ls(hdfs, tmp_dir):
    test_dir = pjoin(tmp_dir, 'ls-test')
    hdfs.mkdir(test_dir)
    n = 2
    to_remove = []
    try:
        for _ in range(n):
            local_path = make_random_file()
            to_remove.append(local_path)
            hdfs_path = pjoin(test_dir, local_path)
            hdfs.put(hdfs_path, local_path)
        assert len(hdfs.ls(test_dir)) == n
    finally:
        for path in to_remove:
            os.remove(path)


def test_size(hdfs, tmp_dir):
    test_dir = pjoin(tmp_dir, 'size-test')

    K = 2048
    path = make_random_file(size=K)
    try:
        hdfs_path = pjoin(test_dir, path)
        hdfs.put(hdfs_path, path)
        assert hdfs.size(hdfs_path) == K

        size_test_dir = sample_nested_directory()
        try:
            hdfs_path = pjoin(test_dir, size_test_dir)
            hdfs.put(hdfs_path, size_test_dir)

            assert hdfs.size(hdfs_path) == K * 7
        finally:
            shutil.rmtree(size_test_dir)
    finally:
        os.remove(path)


def test_put_get_tarfile(hdfs, tmp_dir, tmp_path):
    test_dir = pjoin(tmp_dir, 'tarfile-test')

    dirname = sample_nested_directory()

    try:

        tf_name = tmp_path / f'{dirname}.tar.gz'
        subprocess.check_call(f'tar zc {dirname} > {tf_name}', shell=True)
        randname = util.guid()
        hdfs_path = pjoin(test_dir, randname)
        hdfs.put_tarfile(hdfs_path, tf_name, compression='gzip')

        hdfs.get(hdfs_path, '.')
        _check_directories_equal(osp.join(randname, dirname), dirname)
    finally:
        shutil.rmtree(dirname, ignore_errors=True)
        shutil.rmtree(osp.join(randname, dirname), ignore_errors=True)


def sample_nested_directory():
    K = 2048
    dirname = make_test_directory(files=2, filesize=K)
    nested_dir = osp.join(dirname, util.guid())
    os.mkdir(nested_dir)

    make_test_directory(files=5, filesize=K, directory=nested_dir)

    return dirname


@pytest.fixture
def hdfs_superuser(env, tmp_dir):
    # NOTE: specifying superuser as set in IbisTestEnv
    hdfs = ibis.impala.hdfs_connect(
        host=env.nn_host,
        port=env.webhdfs_port,
        auth_mechanism=env.auth_mechanism,
        verify=env.auth_mechanism not in ['GSSAPI', 'LDAP'],
        user=env.hdfs_superuser,
    )
    hdfs.mkdir(tmp_dir)
    try:
        yield hdfs
    finally:
        hdfs.rmdir(tmp_dir)


@pytest.fixture
def random_hdfs_superuser_file(hdfs_superuser, tmp_dir, random_file):
    local_path = random_file
    remote_path = pjoin(tmp_dir, local_path)
    hdfs_superuser.put(remote_path, local_path)
    return remote_path


def test_chown_owner(hdfs_superuser, tmp_dir, random_hdfs_superuser_file):
    new_owner = 'randomowner'
    path = random_hdfs_superuser_file
    hdfs_superuser.chown(path, new_owner)
    assert hdfs_superuser.status(path)['owner'] == new_owner


def test_chown_group(hdfs_superuser, tmp_dir, random_hdfs_superuser_file):
    new_group = 'randomgroup'
    path = random_hdfs_superuser_file
    hdfs_superuser.chown(path, group=new_group)
    assert hdfs_superuser.status(path)['group'] == new_group


def test_chown_group_directory(
    hdfs_superuser,
    tmp_dir,
    random_hdfs_superuser_file,
):
    new_group = 'randomgroup'
    path = pjoin(tmp_dir, util.guid())
    hdfs_superuser.mkdir(path)
    hdfs_superuser.chown(path, group=new_group)
    assert hdfs_superuser.status(path)['group'] == new_group


def test_chown_owner_directory(
    hdfs_superuser,
    tmp_dir,
    random_hdfs_superuser_file,
):
    new_owner = 'randomowner'
    path = pjoin(tmp_dir, util.guid())
    hdfs_superuser.mkdir(path)
    hdfs_superuser.chown(path, new_owner)
    assert hdfs_superuser.status(path)['owner'] == new_owner


def _check_directories_equal(left, right):
    assert not filecmp.dircmp(left, right).diff_files
