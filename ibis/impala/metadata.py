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


def parse_metadata(descr_table):
    parser = MetadataParser(descr_table)
    return parser.parse()


class MetadataParser(object):

    def __init__(self, table):
        self.table = table

    def parse(self):
        pass


class TableMetadata(object):

    """
    Container for the parsed and wrangled results of DESCRIBE FORMATTED for
    easier Ibis use (and testing).
    """

    @property
    def is_partitioned(self):
        pass
