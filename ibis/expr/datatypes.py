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

import re
import six

import ibis.common as com
import ibis.util as util

if six.PY3:
    from io import StringIO
else:
    from io import BytesIO as StringIO


class Schema(object):

    """
    Holds table schema information
    """

    def __init__(self, names, types):
        if not isinstance(names, list):
            names = list(names)
        self.names = names
        self.types = [validate_type(x) for x in types]

        self._name_locs = dict((v, i) for i, v in enumerate(self.names))

        if len(self._name_locs) < len(self.names):
            raise com.IntegrityError('Duplicate column names')

    def __repr__(self):
        return self._repr()

    def __len__(self):
        return len(self.names)

    def _repr(self):
        buf = StringIO()
        space = 2 + max(len(x) for x in self.names)
        for name, tipo in zip(self.names, self.types):
            buf.write('\n{0}{1}'.format(name.ljust(space), str(tipo)))

        return "ibis.Schema {{{0}\n}}".format(util.indent(buf.getvalue(), 2))

    def __contains__(self, name):
        return name in self._name_locs

    @classmethod
    def from_tuples(cls, values):
        if len(values):
            names, types = zip(*values)
        else:
            names, types = [], []
        return Schema(names, types)

    @classmethod
    def from_dict(cls, values):
        names = list(values.keys())
        types = values.values()
        return Schema(names, types)

    def equals(self, other):
        return ((self.names == other.names) and
                (self.types == other.types))

    def __eq__(self, other):
        return self.equals(other)

    def get_type(self, name):
        return self.types[self._name_locs[name]]

    def append(self, schema):
        names = self.names + schema.names
        types = self.types + schema.types
        return Schema(names, types)


class DataType(object):
    pass


class DecimalType(DataType):
    # Decimal types are parametric, we store the parameters in this object

    def __init__(self, precision, scale):
        self.precision = precision
        self.scale = scale

    def _base_type(self):
        return 'decimal'

    def __repr__(self):
        return ('decimal(precision=%s, scale=%s)'
                % (self.precision, self.scale))

    def __hash__(self):
        return hash((self.precision, self.scale))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        if not isinstance(other, DecimalType):
            return False

        return (self.precision == other.precision and
                self.scale == other.scale)

    def array_ctor(self):
        def constructor(op, name=None):
            from ibis.expr.types import DecimalArray
            return DecimalArray(op, self, name=name)
        return constructor

    def scalar_ctor(self):
        def constructor(op, name=None):
            from ibis.expr.types import DecimalScalar
            return DecimalScalar(op, self, name=name)
        return constructor


class CategoryType(DataType):

    def __init__(self, cardinality=None):
        self.cardinality = cardinality

    def _base_type(self):
        return 'category'

    def __repr__(self):
        card = (self.cardinality if self.cardinality is not None
                else 'unknown')
        return ('category(K=%s)' % card)

    def __hash__(self):
        return hash((self.cardinality))

    def __eq__(self, other):
        if not isinstance(other, CategoryType):
            return False

        return self.cardinality == other.cardinality

    def to_integer_type(self):
        if self.cardinality is None:
            return 'int64'
        elif self.cardinality < (2 ** 7 - 1):
            return 'int8'
        elif self.cardinality < (2 ** 15 - 1):
            return 'int16'
        elif self.cardinality < (2 ** 31 - 1):
            return 'int32'
        else:
            return 'int64'

    def array_ctor(self):
        def constructor(op, name=None):
            from ibis.expr.types import CategoryArray
            return CategoryArray(op, self, name=name)
        return constructor

    def scalar_ctor(self):
        def constructor(op, name=None):
            from ibis.expr.types import CategoryScalar
            return CategoryScalar(op, self, name=name)
        return constructor

# ---------------------------------------------------------------------

_primitive_types = set(['boolean', 'int8', 'int16', 'int32', 'int64',
                        'float', 'double', 'string', 'timestamp',
                        'category'])


def validate_type(t):
    if isinstance(t, DataType):
        return t

    parsed_type = _parse_type(t)
    if parsed_type is not None:
        return parsed_type

    if t not in _primitive_types:
        raise ValueError('Invalid type: %s' % repr(t))
    return t


_DECIMAL_RE = re.compile('decimal\((\d+),[\s]*(\d+)\)')


def _parse_decimal(t):
    m = _DECIMAL_RE.match(t)
    if m:
        precision, scale = m.groups()
        return DecimalType(int(precision), int(scale))

    if t == 'decimal':
        # From the Impala documentation
        return DecimalType(9, 0)


_type_parsers = [
    _parse_decimal
]


def _parse_type(t):
    for parse_fn in _type_parsers:
        parsed = parse_fn(t)
        if parsed is not None:
            return parsed
    return None
