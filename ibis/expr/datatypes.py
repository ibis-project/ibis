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

from collections import namedtuple, OrderedDict

import six

import ibis.common as com
import ibis.util as util


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
        space = 2 + max(map(len, self.names))
        return "ibis.Schema {{{0}\n}}".format(
            util.indent(
                ''.join(
                    '\n{0}{1}'.format(name.ljust(space), str(tipo))
                    for name, tipo in zip(self.names, self.types)
                ),
                2
            )
        )

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

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

    def equals(self, other, cache=None):
        return self.names == other.names and self.types == other.types

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

    def equals(self, other, cache=None):
        if type(self) != type(other):
            return False
        return self.schema.equals(other.schema, cache=cache)

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
        name = self.name.lower()
        if not self.nullable:
            name = '{0}[non-nullable]'.format(name)
        return name

    @property
    def name(self):
        return type(self).__name__

    def equals(self, other, cache=None):
        if isinstance(other, six.string_types):
            other = validate_type(other)

        return (isinstance(other, type(self)) and
                self.nullable == other.nullable)

    def can_implicit_cast(self, other):
        return self.equals(other)

    def scalar_type(self):
        import ibis.expr.types as ir
        return getattr(ir, '{0}Scalar'.format(type(self).__name__))

    def array_type(self):
        import ibis.expr.types as ir
        return getattr(ir, '{0}Column'.format(type(self).__name__))


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


Bounds = namedtuple('Bounds', ('upper', 'lower'))


class Integer(Primitive):

    @property
    def bounds(self):
        exp = self._nbytes * 8 - 1
        lower = -1 << exp
        return Bounds(lower=lower, upper=~lower)

    def can_implicit_cast(self, other):
        return (
            isinstance(other, Integer) and
            (type(self) is Integer or other._nbytes <= self._nbytes)
        )


class String(Variadic):
    pass


class Date(Primitive):
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


class Int16(Integer):

    _nbytes = 2


class Int32(Integer):

    _nbytes = 4


class Int64(Integer):

    _nbytes = 8


class Float(Floating):

    _nbytes = 4


class Double(Floating):

    _nbytes = 8


def parametric(cls):
    type_name = cls.__name__
    array_type_name = '{0}Column'.format(type_name)
    scalar_type_name = '{0}Scalar'.format(type_name)

    def array_type(self):
        def constructor(op, name=None):
            import ibis.expr.types as ir
            return getattr(ir, array_type_name)(op, self, name=name)
        return constructor

    def scalar_type(self):
        def constructor(op, name=None):
            import ibis.expr.types as ir
            return getattr(ir, scalar_type_name)(op, self, name=name)
        return constructor

    cls.array_type = array_type
    cls.scalar_type = scalar_type
    return cls


@parametric
class Decimal(DataType):
    # Decimal types are parametric, we store the parameters in this object

    def __init__(self, precision, scale, nullable=True):
        super(Decimal, self).__init__(nullable=nullable)
        self.precision = precision
        self.scale = scale

    def __repr__(self):
        return '{0}(precision={1:d}, scale={2:d})'.format(
            self.name,
            self.precision,
            self.scale,
        )

    def __str__(self):
        return '{0}({1:d}, {2:d})'.format(
            self.name.lower(),
            self.precision,
            self.scale,
        )

    def __hash__(self):
        return hash((self.precision, self.scale))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (
            isinstance(other, Decimal) and
            self.precision == other.precision and
            self.scale == other.scale
        )

    @classmethod
    def can_implicit_cast(cls, other):
        return isinstance(other, (Floating, Decimal))


@parametric
class Category(DataType):

    def __init__(self, cardinality=None, nullable=True):
        super(Category, self).__init__(nullable=nullable)
        self.cardinality = cardinality

    def __repr__(self):
        if self.cardinality is not None:
            cardinality = self.cardinality
        else:
            cardinality = 'unknown'
        return 'category(K={0})'.format(cardinality)

    def __hash__(self):
        return hash(self.cardinality)

    def __eq__(self, other):
        if not isinstance(other, Category):
            return False

        return self.cardinality == other.cardinality

    def to_integer_type(self):
        cardinality = self.cardinality

        if cardinality is None:
            return int64
        elif cardinality < int8.bounds.upper:
            return int8
        elif cardinality < int16.bounds.upper:
            return int16
        elif cardinality < int32.bounds.upper:
            return int32
        else:
            return int64


@parametric
class Struct(DataType):

    def __init__(self, names, types, nullable=True):
        super(Struct, self).__init__(nullable=nullable)
        self.names = names
        self.types = types

    def __repr__(self):
        return '{0}({1})'.format(
            self.name,
            list(zip(self.names, self.types))
        )

    def __str__(self):
        return '{0}<{1}>'.format(
            self.name.lower(),
            ', '.join(
                '{0}: {1}'.format(n, t) for n, t in zip(self.names, self.types)
            )
        )

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.names == other.names and
                self.types == other.types)

    @classmethod
    def from_tuples(self, pairs):
        return Struct(*map(list, zip(*pairs)))


@parametric
class Array(Variadic):

    def __init__(self, value_type, nullable=True):
        super(Array, self).__init__(nullable=nullable)
        self.value_type = value_type

    def __repr__(self):
        return '{0}({1})'.format(self.name, repr(self.value_type))

    def __str__(self):
        return '{0}<{1}>'.format(self.name.lower(), self.value_type)

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            self.value_type == other.value_type
        )


@parametric
class Enum(DataType):

    def __init__(self, rep_type, value_type, nullable=True):
        super(Enum, self).__init__(nullable=nullable)
        self.rep_type = rep_type
        self.value_type = value_type


@parametric
class Map(DataType):

    def __init__(self, key_type, value_type, nullable=True):
        super(Map, self).__init__(nullable=nullable)
        self.key_type = key_type
        self.value_type = value_type

    def __repr__(self):
        return '{0}({1}, {2})'.format(
            self.name,
            repr(self.key_type),
            repr(self.value_type),
        )

    def __str__(self):
        return '{0}<{1}, {2}>'.format(
            self.name.lower(),
            self.key_type,
            self.value_type,
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            self.key_type == other.key_type and
            self.value_type == other.value_type
        )


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
date = Date()
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
    'date': date,
    'timestamp': timestamp
}


class Tokens(object):
    """Class to hold tokens for lexing
    """
    __slots__ = ()

    ANY = 0
    NULL = 1
    PRIMITIVE = 2
    DECIMAL = 3
    VARCHAR = 4
    CHAR = 5
    ARRAY = 6
    MAP = 7
    STRUCT = 8
    INTEGER = 9
    FIELD = 10
    COMMA = 11
    COLON = 12
    LPAREN = 13
    RPAREN = 14
    LBRACKET = 15
    RBRACKET = 16

    @staticmethod
    def name(value):
        return _token_names[value]


_token_names = dict(
    (getattr(Tokens, n), n)
    for n in dir(Tokens) if n.isalpha() and n.isupper()
)


Token = namedtuple('Token', ('type', 'value'))


_TYPE_RULES = OrderedDict(
    [
        # any, null
        ('(?P<ANY>any)', lambda token: Token(Tokens.ANY, any)),
        ('(?P<NULL>null)', lambda token: Token(Tokens.NULL, null)),
    ] + [
        # primitive types
        (
            '(?P<{}>{})'.format(token.upper(), token),
            lambda token, value=value: Token(Tokens.PRIMITIVE, value)
        ) for token, value in _primitive_types.items()
        if token != 'any' and token != 'null'
    ] + [
        # decimal + complex types
        (
            '(?P<{}>{})'.format(token.upper(), token),
            lambda token, toktype=toktype: Token(toktype, token)
        ) for token, toktype in zip(
            ('decimal', 'varchar', 'char', 'array', 'map', 'struct'),
            (
                Tokens.DECIMAL,
                Tokens.VARCHAR,
                Tokens.CHAR,
                Tokens.ARRAY,
                Tokens.MAP,
                Tokens.STRUCT
            ),
        )
    ] + [
        # numbers, for decimal spec
        (r'(?P<INTEGER>\d+)', lambda token: Token(Tokens.INTEGER, int(token))),

        # struct fields
        (
            r'(?P<FIELD>[a-zA-Z_][a-zA-Z_0-9]*)',
            lambda token: Token(Tokens.FIELD, token)
        ),
        ('(?P<COMMA>,)', lambda token: Token(Tokens.COMMA, token)),
        ('(?P<COLON>:)', lambda token: Token(Tokens.COLON, token)),
        (r'(?P<LPAREN>\()', lambda token: Token(Tokens.LPAREN, token)),
        (r'(?P<RPAREN>\))', lambda token: Token(Tokens.RPAREN, token)),
        ('(?P<LBRACKET><)', lambda token: Token(Tokens.LBRACKET, token)),
        ('(?P<RBRACKET>>)', lambda token: Token(Tokens.RBRACKET, token)),
        (r'(?P<WHITESPACE>\s+)', None),
    ]
)

_TYPE_KEYS = tuple(_TYPE_RULES.keys())
_TYPE_PATTERN = re.compile('|'.join(_TYPE_KEYS), flags=re.IGNORECASE)


def _generate_tokens(pat, text):
    """Generate a sequence of tokens from `text` that match `pat`

    Parameters
    ----------
    pat : compiled regex
        The pattern to use for tokenization
    text : str
        The text to tokenize
    """
    rules = _TYPE_RULES
    keys = _TYPE_KEYS
    groupindex = pat.groupindex
    for m in iter(pat.scanner(text).match, None):
        func = rules[keys[groupindex[m.lastgroup] - 1]]
        if func is not None:
            assert callable(func), 'func must be callable'
            yield func(m.group(m.lastgroup))


class TypeParser(object):
    """A type parser for complex types.

    Parameters
    ----------
    text : str
        The text to parse

    Notes
    -----
    Adapted from David Beazley's and Brian Jones's Python Cookbook
    """

    def __init__(self, text):
        self.text = text
        self.tokens = _generate_tokens(_TYPE_PATTERN, text)
        self.tok = None
        self.nexttok = None

    def _advance(self):
        self.tok, self.nexttok = self.nexttok, next(self.tokens, None)

    def _accept(self, toktype):
        if self.nexttok is not None and self.nexttok.type == toktype:
            self._advance()
            return True
        return False

    def _expect(self, toktype):
        if not self._accept(toktype):
            raise SyntaxError('Expected {0} after {1!r} in {2!r}'.format(
                Tokens.name(toktype),
                self.tok.value,
                self.text,
            ))

    def parse(self):
        self._advance()

        # any and null types cannot be nested
        if self._accept(Tokens.ANY) or self._accept(Tokens.NULL):
            return self.tok.value

        t = self.type()
        if self.nexttok is None:
            return t
        else:
            # additional junk was passed at the end, throw an error
            additional_tokens = []
            while self.nexttok is not None:
                additional_tokens.append(self.nexttok.value)
                self._advance()
            raise SyntaxError(
                'Found additional tokens {0}'.format(additional_tokens)
            )

    def type(self):
        """
        type : primitive
             | decimal
             | array
             | map
             | struct

        primitive : "any"
                  | "null"
                  | "boolean"
                  | "int8"
                  | "int16"
                  | "int32"
                  | "int64"
                  | "float"
                  | "double"
                  | "string"
                  | "timestamp"

        decimal : "decimal"
                | "decimal" "(" integer "," integer ")"

        integer : [0-9]+

        array : "array" "<" type ">"

        map : "map" "<" type "," type ">"

        struct : "struct" "<" field ":" type ("," field ":" type)* ">"

        field : [a-zA-Z_][a-zA-Z_0-9]*
        """
        if self._accept(Tokens.PRIMITIVE):
            return self.tok.value

        elif self._accept(Tokens.DECIMAL):
            if self._accept(Tokens.LPAREN):

                self._expect(Tokens.INTEGER)
                precision = self.tok.value

                self._expect(Tokens.COMMA)

                self._expect(Tokens.INTEGER)
                scale = self.tok.value

                self._expect(Tokens.RPAREN)
            else:
                precision = 9
                scale = 0
            return Decimal(precision, scale)

        elif self._accept(Tokens.VARCHAR) or self._accept(Tokens.CHAR):
            # VARCHAR, VARCHAR(n), CHAR, and CHAR(n) all parse as STRING
            if self._accept(Tokens.LPAREN):
                self._expect(Tokens.INTEGER)
                self._expect(Tokens.RPAREN)
                return string
            return string

        elif self._accept(Tokens.ARRAY):
            self._expect(Tokens.LBRACKET)

            value_type = self.type()

            self._expect(Tokens.RBRACKET)
            return Array(value_type)

        elif self._accept(Tokens.MAP):
            self._expect(Tokens.LBRACKET)

            self._expect(Tokens.PRIMITIVE)
            key_type = self.tok.value

            self._expect(Tokens.COMMA)

            value_type = self.type()

            self._expect(Tokens.RBRACKET)

            return Map(key_type, value_type)

        elif self._accept(Tokens.STRUCT):
            self._expect(Tokens.LBRACKET)

            self._expect(Tokens.FIELD)
            names = [self.tok.value]

            self._expect(Tokens.COLON)

            types = [self.type()]

            while self._accept(Tokens.COMMA):

                self._expect(Tokens.FIELD)
                names.append(self.tok.value)

                self._expect(Tokens.COLON)
                types.append(self.type())

            self._expect(Tokens.RBRACKET)
            return Struct(names, types)
        else:
            raise SyntaxError('Type cannot be parsed: {0}'.format(self.text))


def validate_type(t):
    if isinstance(t, DataType):
        return t
    return TypeParser(t).parse()


def array_type(t):
    # compatibility
    return validate_type(t).array_type()


def scalar_type(t):
    # compatibility
    return validate_type(t).scalar_type()
