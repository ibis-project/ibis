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

from six import StringIO
import pandas as pd


def parse_metadata(descr_table):
    parser = MetadataParser(descr_table)
    return parser.parse()


def _noop(tup):
    return None


def _get_type(converter=None):
    def _converter(tup):
        result = tup[1]
        if converter is not None:
            result = converter(result)
        return result

    return _converter


def _try_timestamp(x):
    try:
        return pd.Timestamp(x)
    except (ValueError, TypeError):
        return x


def _try_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


class MetadataParser(object):

    """
    A simple state-ish machine to parse the results of DESCRIBE FORMATTED
    """

    _info_cleaners = {
        'database': _get_type(),
        'owner': _get_type(),
        'createtime': _get_type(_try_timestamp),
        'lastaccesstime': _get_type(_try_timestamp),
        'protect mode': _get_type(),
        'retention': _get_type(_try_int),
        'location': _get_type(),
        'table type': _get_type()
    }

    def __init__(self, table):
        self.table = table
        self.tuples = list(self.table.itertuples(index=False))

        self.schema = None
        self.partitions = None
        self.info = None
        self.storage = None

    def __repr__(self):
        buf = StringIO()


        return buf.getvalue()

    def _reset(self):
        self.pos = 0

    def _next_tuple(self):
        result = self.tuples[self.pos]
        self.pos += 1
        return result

    def parse(self):
        self._reset()
        self._parse_schema()

    def _parse_schema(self):
        tup = self._next_tuple()
        if 'col_name' not in tup[0]:
            raise ValueError('DESCRIBE FORMATTED did not return '
                             'the expected results: {0}'
                             .format(tup))
        self._next_tuple()

        self.schema = []
        while True:
            tup = self._next_tuple()
            if tup[0].strip() == '':
                break
            self.schema.append((tup[0], tup[1]))

        next_section = self._next_tuple()
        if 'partition' in next_section[0].lower():
            self._parse_partitions()
        else:
            self._parse_info()

    def _parse_partitions(self):
        pass

    def _parse_info(self):
        self.info = {}
        while True:
            tup = self._next_tuple()
            key = tup[0].strip().strip(':').lower()

            if key == '':
                # section is done
                break

            if key == 'table parameters':
                self._parse_table_parameters()
            elif key in self._info_cleaners:
                result = self._info_cleaners[key](tup)
                self.info[key.capitalize()] = result
            else:
                self.info[key.capitalize()] = tup[1]

    def _parse_table_parameters(self):
        params = self.info['Table Parameters'] = {}

    def _parse_storage_info(self):
        pass

    def _parse_storage_desc_params(self):
        pass


class TableMetadata(object):

    """
    Container for the parsed and wrangled results of DESCRIBE FORMATTED for
    easier Ibis use (and testing).
    """

    @property
    def is_partitioned(self):
        pass
