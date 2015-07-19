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

import numpy as np
import pandas as pd
import pandas.core.common as pdcom

import ibis
from ibis.common import IbisTypeError


def guid():
    try:
        from ibis.comms import uuid4_hex
        return uuid4_hex()
    except ImportError:
        from uuid import uuid4
        return uuid4().get_hex()


def bytes_to_uint8_array(val, width=70):
    """
    Formats a byte string for use as a uint8_t* literal in C/C++
    """
    if len(val) == 0:
        return '{}'

    lines = []
    line = '{' + str(ord(val[0]))
    for x in val[1:]:
        token = str(ord(x))
        if len(line) + len(token) > width:
            lines.append(line + ',')
            line = token
        else:
            line += ',%s' % token
    lines.append(line)
    return '\n'.join(lines) + '}'


def unique_by_key(values, key):
    id_to_table = {}
    for x in values:
        id_to_table[key(x)] = x
    return id_to_table.values()


def indent(text, spaces):
    block = ' ' * spaces
    return '\n'.join(block + x for x in text.split('\n'))


def any_of(values, t):
    for x in values:
        if isinstance(x, t):
            return True
    return False


def all_of(values, t):
    for x in values:
        if not isinstance(x, t):
            return False
    return True


def promote_list(val):
    if not isinstance(val, list):
        val = [val]
    return val


class IbisSet(object):

    def __init__(self, keys=None):
        self.keys = keys or []

    @classmethod
    def from_list(cls, keys):
        return IbisSet(keys)

    def __contains__(self, obj):
        for other in self.keys:
            if obj.equals(other):
                return True
        return False

    def add(self, obj):
        self.keys.append(obj)


class IbisMap(object):

    def __init__(self):
        self.keys = []
        self.values = []

    def __contains__(self, obj):
        for other in self.keys:
            if obj.equals(other):
                return True
        return False

    def set(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def get(self, key):
        for k, v in zip(self.keys, self.values):
            if key.equals(k):
                return v
        raise KeyError(key)


def pandas_col_to_ibis_type(col):
    dty = col.dtype

    # datetime types
    if pdcom.is_datetime64_dtype(dty):
        if pdcom.is_datetime64_ns_dtype(dty):
            return 'timestamp'
        else:
            raise IbisTypeError(
                "Column {0} has dtype {1}, which is datetime64-like but does "
                "not use nanosecond units".format(col.name, dty))
    if pdcom.is_timedelta64_dtype(dty):
        print("Warning: encoding a timedelta64 as an int64")
        return 'int64'

    if pdcom.is_categorical_dtype(dty):
        return 'category'

    if pdcom.is_bool_dtype(dty):
        return 'boolean'

    # simple numerical types
    if issubclass(dty.type, np.int8):
        return 'int8'
    if issubclass(dty.type, np.int16):
        return 'int16'
    if issubclass(dty.type, np.int32):
        return 'int32'
    if issubclass(dty.type, np.int64):
        return 'int64'
    if issubclass(dty.type, np.float32):
        return 'float'
    if issubclass(dty.type, np.float64):
        return 'double'
    if issubclass(dty.type, np.uint8):
        return 'int16'
    if issubclass(dty.type, np.uint16):
        return 'int32'
    if issubclass(dty.type, np.uint32):
        return 'int64'
    if issubclass(dty.type, np.uint64):
        raise IbisTypeError("Column {0} is an unsigned int64".format(col.name))

    if pdcom.is_object_dtype(dty):
        # TODO: overly broad?
        return 'string'

    raise IbisTypeError("Column {0} is dtype {1}".format(col.name, dty))


def pandas_to_ibis_schema(frame):
    # no analog for decimal in pandas
    pairs = []
    for col_name in frame:
        ibis_type = pandas_col_to_ibis_type(frame[col_name])
        pairs.append((col_name, ibis_type))
    return ibis.schema(pairs)
