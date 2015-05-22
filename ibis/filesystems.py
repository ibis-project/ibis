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

import hdfs

import ibis.common as com


class HDFS(object):

    """
    Interface class to HDFS for ibis that abstracts away (and protects
    user/developer against) various 3rd party library API differences.
    """

    def path_exists(self, path):
        try:
            self.client.status(path)
            return True
        except Exception:
            return False

    def head(self, hdfs_path, nbytes=1024, offset=0):
        raise NotImplementedError

    def get(self, hdfs_path, local_path, overwrite=False):
        raise NotImplementedError

    def put(self, hdfs_path, local_path, overwrite=False, **kwargs):
        raise NotImplementedError

    def write(self, hdfs_path, buf, overwrite=False, blocksize=None,
              replication=None, buffersize=None):
        raise NotImplementedError

    def mkdir(self, path, create_parent=False):
        pass

    def ls(self, hdfs_path):
        raise NotImplementedError

    def tail(self, hdfs_path, nbytes=1024):
        raise NotImplementedError

    def delete(self, hdfs_path):
        pass

    def rmdir(self, path):
        self.client.delete(path, recursive=True)

    def _find_any_file(self, hdfs_dir):
        contents = self.ls(hdfs_dir)
        for filename, meta in contents:
            if meta['type'].lower() == 'file':
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

    def ls(self, hdfs_path):
        return self.client.list(hdfs_path)

    def head(self, hdfs_path, nbytes=1024, offset=0):
        gen = self.client.read(hdfs_path, offset=offset, length=nbytes)
        return ''.join(gen)

    def put(self, hdfs_path, local_path, overwrite=False, **kwargs):
        """

        Parameters
        ----------

        Other keywords forwarded to .write API.
        """
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
