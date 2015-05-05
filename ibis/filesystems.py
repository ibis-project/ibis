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


class HDFSInterface(object):

    """

    """

    def __init__(self, client):
        self.client = client

    def delete(self, hdfs_path):
        pass

    def put(self, local_path, hdfs_path):
        pass

    def mkdir(self, path, create_parent=False):
        pass

    def rmdir(self, path):
        pass



def connect(host, port=8020, hadoop_version=9):
    from snakebite.client import Client
    client = Client(host, port=port, hadoop_version=hadoop_version)
    return HDFSInterface(client)


def connect_ha(namenodes):
    raise NotImplementedError
