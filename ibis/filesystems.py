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

import os
import posixpath

import ibis.common as com
import ibis.util as util


def implements(f):
    def decorator(g):
        g.__doc__ = f.__doc__
        return g
    return decorator


class HDFS(object):

    """
    Interface class to HDFS for ibis that abstracts away (and protects
    user/developer against) various 3rd party library API differences.
    """
    def log(self, message):
        print(message)

    def exists(self, path):
        raise NotImplementedError

    def head(self, hdfs_path, nbytes=1024, offset=0):
        raise NotImplementedError

    def get(self, hdfs_path, local_path, overwrite=False):
        raise NotImplementedError

    def put(self, hdfs_path, local_path, overwrite=False, verbose=None,
            **kwargs):
        """

        """
        raise NotImplementedError

    def write(self, hdfs_path, buf, overwrite=False, blocksize=None,
              replication=None, buffersize=None):
        raise NotImplementedError

    def mkdir(self, path, create_parent=False):
        pass

    def ls(self, hdfs_path):
        """
        Return contents of directory

        Parameters
        ----------
        hdfs_path : string
        """
        raise NotImplementedError

    def tail(self, hdfs_path, nbytes=1024):
        raise NotImplementedError

    def rm(self, path):
        return self.delete(path)

    def rmdir(self, path):
        self.client.delete(path, recursive=True)

    def find_any_file(self, hdfs_dir):
        contents = self.ls(hdfs_dir)

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

    @implements(HDFS.exists)
    def exists(self, path):
        try:
            self.client.status(path)
            return True
        except Exception:
            return False

    @implements(HDFS.ls)
    def ls(self, hdfs_path):
        return self.client.list(hdfs_path)

    @implements(HDFS.mkdir)
    def mkdir(self, dir_path, create_parent=False):
        # ugh, see #252

        # create a temporary file, then delete it
        dummy = posixpath.join(dir_path, util.guid())
        self.client.write(dummy, '')
        self.client.delete(dummy)

    def delete(self, hdfs_path, recursive=False):
        """

        """
        return self.client.delete(hdfs_path, recursive=recursive)

    def head(self, hdfs_path, nbytes=1024, offset=0):
        gen = self.client.read(hdfs_path, offset=offset, length=nbytes)
        return ''.join(gen)

    @implements(HDFS.put)
    def put(self, hdfs_path, local_path, overwrite=False, verbose=None,
            **kwargs):
        if os.path.isdir(local_path):
            for dirpath, dirnames, filenames in os.walk(local_path):
                rel_dir = os.path.relpath(dirpath, local_path)
                if rel_dir == '.':
                    rel_dir = ''
                for fpath in filenames:
                    abs_path = os.path.join(dirpath, fpath)
                    rel_hdfs_path = posixpath.join(hdfs_path, rel_dir, fpath)
                    self.put(rel_hdfs_path, abs_path, overwrite=overwrite,
                             verbose=verbose, **kwargs)
        else:
            if verbose:
                self.log('Writing local {} to HDFS {}'.format(local_path,
                                                              hdfs_path))
            self.client.upload(hdfs_path, local_path,
                               overwrite=overwrite, **kwargs)

    def write(self, hdfs_path, buf, overwrite=False, blocksize=None,
              replication=None, buffersize=None):
        """
        Write a buffer-like object to indicated HDFS path

        Parameters
        ----------
        """
        self.client.write(buf, hdfs_path, overwrite=overwrite,
                          blocksize=blocksize, replication=replication,
                          buffersize=buffersize)


def connect(host, port=8020, hadoop_version=9):
    from snakebite.client import Client
    client = Client(host, port=port, hadoop_version=hadoop_version)
    return HDFS(client)


def connect_ha(namenodes):
    raise NotImplementedError
