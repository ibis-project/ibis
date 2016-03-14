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


def _item_converter(i):
    def _get_item(converter=None):
        def _converter(tup):
            result = tup[i]
            if converter is not None:
                result = converter(result)
            return result

        return _converter

    return _get_item

_get_type = _item_converter(1)
_get_comment = _item_converter(2)


def _try_timestamp(x):
    try:
        ts = pd.Timestamp(x)
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        return x


def _try_unix_timestamp(x):
    try:
        ts = pd.Timestamp.fromtimestamp(int(x))
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        return x


def _try_boolean(x):
    try:
        x = x.lower()
        if x in ('true', 'yes'):
            return True
        elif x in ('false', 'no'):
            return False
        return x
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

    def __init__(self, table):
        self.table = table
        self.tuples = list(self.table.itertuples(index=False))

    def _reset(self):
        self.pos = 0
        self.schema = None
        self.partitions = None
        self.info = None
        self.storage = None

    def _next_tuple(self):
        if self.pos == len(self.tuples):
            raise StopIteration

        result = self.tuples[self.pos]
        self.pos += 1
        return result

    def parse(self):
        self._reset()
        self._parse()

        return TableMetadata(self.schema, self.info, self.storage,
                             partitions=self.partitions)

    def _parse(self):
        self.schema = self._parse_schema()

        next_section = self._next_tuple()
        if 'partition' in next_section[0].lower():
            self._parse_partitions()
        else:
            self._parse_info()

    def _parse_partitions(self):
        self.partitions = self._parse_schema()

        next_section = self._next_tuple()
        if 'table information' not in next_section[0].lower():
            raise ValueError('Table information not present')

        self._parse_info()

    def _parse_schema(self):
        tup = self._next_tuple()
        if 'col_name' not in tup[0]:
            raise ValueError('DESCRIBE FORMATTED did not return '
                             'the expected results: {0}'
                             .format(tup))
        self._next_tuple()

        # Use for both main schema and partition schema (if any)
        schema = []
        while True:
            tup = self._next_tuple()
            if tup[0].strip() == '':
                break
            schema.append((tup[0], tup[1]))

        return schema

    def _parse_info(self):
        self.info = {}
        while True:
            tup = self._next_tuple()
            orig_key = tup[0].strip(':')
            key = _clean_param_name(tup[0])

            if key == '' or key.startswith('#'):
                # section is done
                break

            if key == 'table parameters':
                self._parse_table_parameters()
            elif key in self._info_cleaners:
                result = self._info_cleaners[key](tup)
                self.info[orig_key] = result
            else:
                self.info[orig_key] = tup[1]

        if 'storage information' not in key:
            raise ValueError('Storage information not present')

        self._parse_storage_info()

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

    def _parse_table_parameters(self):
        params = self._parse_nested_params(self._table_param_cleaners)
        self.info['Table Parameters'] = params

    _table_param_cleaners = {
        'external': _try_boolean,
        'column_stats_accurate': _try_boolean,
        'numfiles': _try_int,
        'totalsize': _try_int,
        'stats_generated_via_stats_task': _try_boolean,
        'numrows': _try_int,
        'transient_lastddltime': _try_unix_timestamp,
    }

    def _parse_storage_info(self):
        self.storage = {}
        while True:
            # end of the road
            try:
                tup = self._next_tuple()
            except StopIteration:
                break

            orig_key = tup[0].strip(':')
            key = _clean_param_name(tup[0])

            if key == '' or key.startswith('#'):
                # section is done
                break

            if key == 'storage desc params':
                self._parse_storage_desc_params()
            elif key in self._storage_cleaners:
                result = self._storage_cleaners[key](tup)
                self.storage[orig_key] = result
            else:
                self.storage[orig_key] = tup[1]

    _storage_cleaners = {
        'compressed': _get_type(_try_boolean),
        'num buckets': _get_type(_try_int),
    }

    def _parse_storage_desc_params(self):
        params = self._parse_nested_params(self._storage_param_cleaners)
        self.storage['Desc Params'] = params

    _storage_param_cleaners = {}

    def _parse_nested_params(self, cleaners):
        params = {}
        while True:
            try:
                tup = self._next_tuple()
            except StopIteration:
                break
            if pd.isnull(tup[1]):
                break

            key, value = tup[1:]

            if key.lower() in cleaners:
                cleaner = cleaners[key.lower()]
                value = cleaner(value)
            params[key] = value

        return params


def _clean_param_name(x):
    return x.strip().strip(':').lower()


def _get_meta(attr, key):
    @property
    def f(self):
        data = getattr(self, attr)
        if isinstance(key, list):
            result = data
            for k in key:
                if k not in result:
                    raise KeyError(k)
                result = result[k]
            return result
        else:
            return data[key]
    return f


class TableMetadata(object):

    """
    Container for the parsed and wrangled results of DESCRIBE FORMATTED for
    easier Ibis use (and testing).
    """
    def __init__(self, schema, info, storage, partitions=None):
        self.schema = schema
        self.info = info
        self.storage = storage
        self.partitions = partitions

    def __repr__(self):
        import pprint

        # Quick and dirty for now
        buf = StringIO()
        buf.write(str(type(self)))
        buf.write('\n')

        data = {
            'schema': self.schema,
            'info': self.info,
            'storage info': self.storage
        }
        if self.partitions is not None:
            data['partition schema'] = self.partitions

        pprint.pprint(data, stream=buf)

        return buf.getvalue()

    @property
    def is_partitioned(self):
        return self.partitions is not None

    create_time = _get_meta('info', 'CreateTime')
    location = _get_meta('info', 'Location')
    owner = _get_meta('info', 'Owner')
    num_rows = _get_meta('info', ['Table Parameters', 'numRows'])
    hive_format = _get_meta('storage', 'InputFormat')

    tbl_properties = _get_meta('info', 'Table Parameters')
    serde_properties = _get_meta('storage', 'Desc Params')


class TableInfo(object):
    pass


class TableStorageInfo(object):
    pass
