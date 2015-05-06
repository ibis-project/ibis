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


class HDFS(object):

    """
    Interface class to HDFS for ibis that abstracts away (and protects
    user/developer against) various 3rd party library API differences.
    """

    def __init__(self, host, port, params=None, protocol='webhdfs'):
        self.host = host
        self.port = port
        self.params = params
        self.protocol = protocol

        if self.protocol == 'webhdfs':
            url = 'http://{}:{}'.format(self.host, self.port)
            self.client = hdfs.client.Client(url, params=self.params)
        else:
            raise NotImplementedError

    def path_exists(self, path):
        try:
            self.client.status(path)
            return True
        except Exception:
            return False

    def ls(self, hdfs_path):
        if self.protocol == 'webhdfs':
            return self.client.list(hdfs_path)

    def delete(self, hdfs_path):
        pass

    def head(self, hdfs_path, nbytes=1024, offset=0):
        if self.protocol == 'webhdfs':
            gen = self.client.read(hdfs_path, offset=offset, length=nbytes)
            return ''.join(gen)

    def tail(self, hdfs_path, nbytes=1024):
        raise NotImplementedError

    def get(self, hdfs_path, local_path, overwrite=False):
        """
        Download a file from HDFS to local
        """
        pass

    def put(self, hdfs_path, local_path, overwrite=False, **kwargs):
        """

        Parameters
        ----------

        Other keywords forwarded to .write API.
        """
        if self.protocol == 'webhdfs':
            self.client.upload(hdfs_path, local_path,
                               overwrite=overwrite, **kwargs)

    def write(self, hdfs_path, buf, overwrite=False, blocksize=None,
              replication=None, buffersize=None):
        """
        Write a buffer-like object to indicated HDFS path

        Parameters
        ----------
        """
        if self.protocol == 'webhdfs':
            self.client.write(buf, hdfs_path, overwrite=overwrite,
                              blocksize=blocksize, replication=replication,
                              buffersize=buffersize)

    def mkdir(self, path, create_parent=False):
        pass

    def rmdir(self, path):
        self.client.delete(path, recursive=True)



def connect(host, port=8020, hadoop_version=9):
    from snakebite.client import Client
    client = Client(host, port=port, hadoop_version=hadoop_version)
    return HDFS(client)


def connect_ha(namenodes):
    raise NotImplementedError
