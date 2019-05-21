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

# This file may adapt small portions of https://github.com/mtth/hdfs (MIT
# license), see the LICENSES directory.

import posixpath

import six

from ibis.config import options
from ibis.util import implements
import ibis.common as com


class HDFSError(com.IbisError):
    pass


class HDFS(object):

    """
    Interface class to HDFS for ibis that abstracts away (and protects
    user/developer against) various 3rd party library API differences.
    """
    def log(self, message):
        print(message)

    def exists(self, path):
        raise NotImplementedError

    def status(self, path):
        raise NotImplementedError

    def chmod(self, hdfs_path, permissions):
        """
        Change permissions of a file of directory

        Parameters
        ----------
        hdfs_path : string
          Directory or path
        permissions : string
          Octal permissions string
        """
        raise NotImplementedError

    def chown(self, hdfs_path, owner=None, group=None):
        """
        Change owner (and/or group) of a file or directory

        Parameters
        ----------
        hdfs_path : string
          Directory or path
        owner : string, optional
          Name of owner
        group : string, optional
          Name of group
        """
        raise NotImplementedError

    def head(self, hdfs_path, nbytes=1024, offset=0):
        """
        Retrieve the requested number of bytes from a file

        Parameters
        ----------
        hdfs_path : string
          Absolute HDFS path
        nbytes : int, default 1024 (1K)
          Number of bytes to retrieve
        offset : int, default 0
          Number of bytes at beginning of file to skip before retrieving data

        Returns
        -------
        head_data : bytes
        """
        raise NotImplementedError

    def get(self, hdfs_path, local_path='.', overwrite=False):
        """
        Download remote file or directory to the local filesystem

        Parameters
        ----------
        hdfs_path : string
        local_path : string, default '.'

        Further keyword arguments passed down to any internal API used.

        Returns
        -------
        written_path : string
          The path to the written file or directory
        """
        raise NotImplementedError

    def put(self, hdfs_path, resource, overwrite=False, verbose=None,
            **kwargs):
        """
        Write file or directory to HDFS

        Parameters
        ----------
        hdfs_path : string
          Directory or path
        resource : string or buffer-like
          Relative or absolute path to local resource, or a file-like object
        overwrite : boolean, default False
        verbose : boolean, default ibis options.verbose

        Further keyword arguments passed down to any internal API used.

        Returns
        -------
        written_path : string
          The path to the written file or directory
        """
        raise NotImplementedError

    def put_tarfile(self, hdfs_path, local_path, compression='gzip',
                    verbose=None, overwrite=False):
        """
        Write contents of tar archive to HDFS directly without having to
        decompress it locally first

        Parameters
        ----------
        hdfs_path : string
        local_path : string
        compression : {'gzip', 'bz2', None}
        overwrite : boolean, default False
        verbose : boolean, default None (global default)
        """
        import tarfile
        modes = {
            None: 'r',
            'gzip': 'r:gz',
            'bz2': 'r:bz2'
        }

        if compression not in modes:
            raise ValueError('Invalid compression type {0}'
                             .format(compression))
        mode = modes[compression]

        tf = tarfile.open(local_path, mode=mode)
        for info in tf:
            if not info.isfile():
                continue

            buf = tf.extractfile(info)
            abspath = posixpath.join(hdfs_path, info.path)
            self.put(abspath, buf, verbose=verbose, overwrite=overwrite)

    def put_zipfile(self, hdfs_path, local_path):
        raise NotImplementedError

    def write(self, hdfs_path, buf, overwrite=False, blocksize=None,
              replication=None, buffersize=None):
        raise NotImplementedError

    def mkdir(self, path):
        pass

    def ls(self, hdfs_path, status=False):
        """
        Return contents of directory

        Parameters
        ----------
        hdfs_path : string
        """
        raise NotImplementedError

    def size(self, hdfs_path):
        """
        Return total size of file or directory

        Parameters
        ----------
        size : int
        """
        raise NotImplementedError

    def tail(self, hdfs_path, nbytes=1024):
        raise NotImplementedError

    def mv(self, hdfs_path_src, hdfs_path_dest, overwrite=True):
        """
        Move hdfs_path_src to hdfs_path_dest

        Parameters
        ----------
        overwrite : boolean, default True
          Overwrite hdfs_path_dest if it exists.
        """
        raise NotImplementedError

    def cp(self, hdfs_path_src, hdfs_path_dest):
        raise NotImplementedError

    def rm(self, path):
        """
        Delete a single file
        """
        return self.delete(path)

    def rmdir(self, path):
        """
        Delete a directory and all its contents
        """
        self.client.delete(path, recursive=True)

    def _find_any_file(self, hdfs_dir):
        contents = self.ls(hdfs_dir, status=True)

        def valid_filename(name):
            head, tail = posixpath.split(name)

            tail = tail.lower()
            return (not tail.endswith('.tmp') and
                    not tail.endswith('.copying') and
                    not tail.startswith('_') and
                    not tail.startswith('.'))

        for filename, meta in contents:
            if meta['type'].lower() == 'file' and valid_filename(filename):
                return filename
        raise com.IbisError('No files found in the passed directory')


class WebHDFS(HDFS):

    """
    A WebHDFS-based interface to HDFS using the HDFSCli library
    """

    def __init__(self, client):
        self.client = client

    @property
    def protocol(self):
        return 'webhdfs'

    def status(self, path):
        """
        Retrieve HDFS metadata for path
        """
        return self.client.status(path)

    @implements(HDFS.chmod)
    def chmod(self, path, permissions):
        self.client.set_permission(path, permissions)

    @implements(HDFS.chown)
    def chown(self, path, owner=None, group=None):
        self.client.set_owner(path, owner, group)

    @implements(HDFS.exists)
    def exists(self, path):
        return not self.client.status(path, strict=False) is None

    @implements(HDFS.ls)
    def ls(self, hdfs_path, status=False):
        return self.client.list(hdfs_path, status=status)

    @implements(HDFS.mkdir)
    def mkdir(self, dir_path):
        self.client.makedirs(dir_path)

    @implements(HDFS.size)
    def size(self, hdfs_path):
        return self.client.content(hdfs_path)['length']

    @implements(HDFS.mv)
    def mv(self, hdfs_path_src, hdfs_path_dest, overwrite=True):
        if overwrite and self.exists(hdfs_path_dest):
            if self.status(hdfs_path_dest)['type'] == 'FILE':
                self.rm(hdfs_path_dest)
        return self.client.rename(hdfs_path_src, hdfs_path_dest)

    def delete(self, hdfs_path, recursive=False):
        """
        Delete a file.
        """
        return self.client.delete(hdfs_path, recursive=recursive)

    @implements(HDFS.head)
    def head(self, hdfs_path, nbytes=1024, offset=0):
        _reader = self.client.read(hdfs_path, offset=offset, length=nbytes)
        with _reader as reader:
            return reader.read()

    @implements(HDFS.put)
    def put(self, hdfs_path, resource, overwrite=False, verbose=None,
            **kwargs):
        verbose = verbose or options.verbose
        if isinstance(resource, six.string_types):
            # `resource` is a path.
            return self.client.upload(hdfs_path, resource, overwrite=overwrite,
                                      **kwargs)
        else:
            # `resource` is a file-like object.
            hdfs_path = self.client.resolve(hdfs_path)
            self.client.write(hdfs_path, data=resource, overwrite=overwrite,
                              **kwargs)
            return hdfs_path

    @implements(HDFS.get)
    def get(self, hdfs_path, local_path, overwrite=False, verbose=None,
            **kwargs):
        verbose = verbose or options.verbose
        return self.client.download(hdfs_path, local_path, overwrite=overwrite,
                                    **kwargs)
