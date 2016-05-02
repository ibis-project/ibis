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

import ibis.expr.types as ir
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

    def __iter__(self):
        return iter(self.names)

    def _repr(self):
        buf = StringIO()
        space = 2 + max(len(x) for x in self.names)
        for name, tipo in zip(self.names, self.types):
            buf.write('\n{0}{1}'.format(name.ljust(space), str(tipo)))

        return "ibis.Schema {{{0}\n}}".format(util.indent(buf.getvalue(), 2))

    def __contains__(self, name):
        return name in self._name_locs

    def __getitem__(self, name):
        return self.types[self._name_locs[name]]

    def delete(self, names_to_delete):
        for name in names_to_delete:
            if name not in self:
                raise KeyError(name)

        new_names, new_types = [], []
        for name, type_ in zip(self.names, self.types):
            if name in names_to_delete:
                continue
            new_names.append(name)
            new_types.append(type_)

        return Schema(new_names, new_types)

    @classmethod
    def from_tuples(cls, values):
        if not isinstance(values, (list, tuple)):
            values = list(values)

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

    def items(self):
        return zip(self.names, self.types)


class HasSchema(object):

    """
    Base class representing a structured dataset with a well-defined
    schema.

    Base implementation is for tables that do not reference a particular
    concrete dataset or database table.
    """

    def __init__(self, schema, name=None):
        assert isinstance(schema, Schema)
        self._schema = schema
        self._name = name

    def __repr__(self):
        return self._repr()

    def _repr(self):
        return "%s(%s)" % (type(self).__name__, repr(self.schema))

    @property
    def schema(self):
        return self._schema

    def get_schema(self):
        return self._schema

    def has_schema(self):
        return True

    @property
    def name(self):
        return self._name

    def equals(self, other):
        if type(self) != type(other):
            return False
        return self.schema.equals(other.schema)

    def root_tables(self):
        return [self]


class DataType(object):

    def __init__(self, nullable=True):
        self.nullable = nullable

    def __call__(self, nullable=True):
        return self._factory(nullable=nullable)

    def _factory(self, nullable=True):
        return type(self)(nullable=nullable)

    def __eq__(self, other):
        return self.equals(other)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        name = self.name()
        if not self.nullable:
            name = '{0}[non-nullable]'.format(name)
        return name

    def name(self):
        return type(self).__name__.lower()

    def equals(self, other):
        if isinstance(other, six.string_types):
            other = validate_type(other)

        return (isinstance(other, type(self)) and
                self.nullable == other.nullable)

    def can_implicit_cast(self, other):
        return self.equals(other)

    def scalar_type(self):
        name = type(self).__name__
        return getattr(ir, '{0}Scalar'.format(name))

    def array_type(self):
        name = type(self).__name__
        return getattr(ir, '{0}Array'.format(name))


class Any(DataType):
    pass


class Primitive(DataType):
    pass


class Null(DataType):
    pass


class Variadic(DataType):
    pass


class Boolean(Primitive):
    pass


class Integer(Primitive):

    def can_implicit_cast(self, other):
        if isinstance(other, Integer):
            return ((type(self) == Integer) or
                    (other._nbytes <= self._nbytes))
        else:
            return False


class String(Variadic):
    pass


class Timestamp(Primitive):
    pass


class SignedInteger(Integer):
    pass


class Floating(Primitive):

    def can_implicit_cast(self, other):
        if isinstance(other, Integer):
            return True
        elif isinstance(other, Floating):
            # return other._nbytes <= self._nbytes
            return True
        else:
            return False


class Int8(Integer):

    _nbytes = 1
    bounds = (-128, 127)


class Int16(Integer):

    _nbytes = 2
    bounds = (-32768, 32767)


class Int32(Integer):

    _nbytes = 4
    bounds = (-2147483648, 2147483647)


class Int64(Integer):

    _nbytes = 8
    bounds = (-9223372036854775808, 9223372036854775807)


class Float(Floating):

    _nbytes = 4


class Double(Floating):

    _nbytes = 8


class Decimal(DataType):
    # Decimal types are parametric, we store the parameters in this object

    def __init__(self, precision, scale, nullable=True):
        self.precision = precision
        self.scale = scale
        DataType.__init__(self, nullable=nullable)

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
        if not isinstance(other, Decimal):
            return False

        return (self.precision == other.precision and
                self.scale == other.scale)

    @classmethod
    def can_implicit_cast(cls, other):
        return isinstance(other, (Floating, Decimal))

    def array_type(self):
        def constructor(op, name=None):
            from ibis.expr.types import DecimalArray
            return DecimalArray(op, self, name=name)
        return constructor

    def scalar_type(self):
        def constructor(op, name=None):
            from ibis.expr.types import DecimalScalar
            return DecimalScalar(op, self, name=name)
        return constructor


class Category(DataType):

    def __init__(self, cardinality=None, nullable=True):
        self.cardinality = cardinality
        DataType.__init__(self, nullable=nullable)

    def _base_type(self):
        return 'category'

    def __repr__(self):
        card = (self.cardinality if self.cardinality is not None
                else 'unknown')
        return ('category(K=%s)' % card)

    def __hash__(self):
        return hash(self.cardinality)

    def __eq__(self, other):
        if not isinstance(other, Category):
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

    def array_type(self):
        def constructor(op, name=None):
            from ibis.expr.types import CategoryArray
            return CategoryArray(op, self, name=name)
        return constructor

    def scalar_type(self):
        def constructor(op, name=None):
            from ibis.expr.types import CategoryScalar
            return CategoryScalar(op, self, name=name)
        return constructor


class Struct(DataType):

    def __init__(self, names, types, nullable=True):
        DataType.__init__(self, nullable=nullable)


class Array(Variadic):

    def __init__(self, value_type, nullable=True):
        Variadic.__init__(self, nullable=nullable)


class Enum(DataType):

    def __init__(self, rep_type, value_type, nullable=True):
        DataType.__init__(self, nullable=nullable)


class Map(DataType):

    def __init__(self, key_type, value_type, nullable=True):
        DataType.__init__(self, nullable=nullable)


# ---------------------------------------------------------------------


any = Any()
null = Null()
boolean = Boolean()
int_ = Integer()
int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
float = Float()
double = Double()
string = String()
timestamp = Timestamp()


_primitive_types = {
    'any': any,
    'null': null,
    'boolean': boolean,
    'int8': int8,
    'int16': int16,
    'int32': int32,
    'int64': int64,
    'float': float,
    'double': double,
    'string': string,
    'timestamp': timestamp
}


def validate_type(t):
    if isinstance(t, DataType):
        return t

    parsed_type = _parse_type(t)
    if parsed_type is not None:
        return parsed_type

    if t in _primitive_types:
        return _primitive_types[t]
    else:
        raise ValueError('Invalid type: %s' % repr(t))


_DECIMAL_RE = re.compile('decimal\((\d+),[\s]*(\d+)\)')


def _parse_decimal(t):
    m = _DECIMAL_RE.match(t)
    if m:
        precision, scale = m.groups()
        return Decimal(int(precision), int(scale))

    if t == 'decimal':
        # From the Impala documentation
        return Decimal(9, 0)


_type_parsers = [
    _parse_decimal
]


def _parse_type(t):
    for parse_fn in _type_parsers:
        parsed = parse_fn(t)
        if parsed is not None:
            return parsed
    return None


def array_type(t):
    # compatibility
    return validate_type(t).array_type()


def scalar_type(t):
    # compatibility
    return validate_type(t).scalar_type()
