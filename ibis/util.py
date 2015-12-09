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

import types
import ibis.compat as compat

from ibis.config import options


def guid():
    try:
        from ibis.comms import uuid4_hex
        return uuid4_hex()
    except ImportError:
        from uuid import uuid4
        guid = uuid4()
        return guid.hex if compat.PY3 else guid.get_hex()


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
    return compat.dict_values(id_to_table)


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


def is_function(v):
    return isinstance(v, (types.FunctionType, types.LambdaType))


def adjoin(space, *lists):
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    Brought over from from pandas
    """
    out_lines = []
    newLists = []
    lengths = [max(map(len, x)) + space for x in lists[:-1]]

    # not the last one
    lengths.append(max(map(len, lists[-1])))

    maxLen = max(map(len, lists))
    for i, lst in enumerate(lists):
        nl = [x.ljust(lengths[i]) for x in lst]
        nl.extend([' ' * lengths[i]] * (maxLen - len(lst)))
        newLists.append(nl)
    toJoin = zip(*newLists)
    for lines in toJoin:
        out_lines.append(_join_unicode(lines))
    return _join_unicode(out_lines, sep='\n')


def _join_unicode(lines, sep=''):
    try:
        return sep.join(lines)
    except UnicodeDecodeError:
        sep = compat.unicode_type(sep)
        return sep.join([x.decode('utf-8') if isinstance(x, str) else x
                         for x in lines])


def deprecate(f, message):
    def g(*args, **kwargs):
        print(message)
        return f(*args, **kwargs)
    return g


def to_stdout(x):
    print(x)


def log(msg):
    if options.verbose:
        (options.verbose_log or to_stdout)(msg)


class cache_readonly(object):

    def __init__(self, func=None, allow_setting=False):
        if func is not None:
            self.func = func
            self.name = func.__name__
        self.allow_setting = allow_setting

    def __call__(self, func, doc=None):
        self.func = func
        self.name = func.__name__
        return self

    def __get__(self, obj, typ):
        # Get the cache or set a default one if needed

        cache = getattr(obj, '_cache', None)
        if cache is None:
            try:
                cache = obj._cache = {}
            except (AttributeError):
                return

        if self.name in cache:
            val = cache[self.name]
        else:
            val = self.func(obj)
            cache[self.name] = val
        return val

    def __set__(self, obj, value):
        if not self.allow_setting:
            raise Exception("cannot set values for [%s]" % self.name)

        # Get the cache or set a default one if needed
        cache = getattr(obj, '_cache', None)
        if cache is None:
            try:
                cache = obj._cache = {}
            except (AttributeError):
                return

        cache[self.name] = value


def approx_equal(a, b, eps):
    assert abs(a - b) < eps


def implements(f):
    def decorator(g):
        g.__doc__ = f.__doc__
        return g
    return decorator
