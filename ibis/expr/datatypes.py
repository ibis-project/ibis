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
import toolz
import datetime
import itertools
import functools
import numpy as np

from collections import namedtuple, OrderedDict
from multipledispatch import Dispatcher

import ibis
import ibis.common as com
import ibis.util as util

from ibis.compat import builtins, PY2


class Schema(object):

    """An object for holding table schema information, i.e., column names and
    types.

    Parameters
    ----------
    names : Sequence[str]
        A sequence of ``str`` indicating the name of each column.
    types : Sequence[DataType]
        A sequence of :class:`ibis.expr.datatypes.DataType` objects
        representing type of each column.
    """

    __slots__ = 'names', 'types', '_name_locs'

    def __init__(self, names, types):
        if not isinstance(names, list):
            names = list(names)

        self.names = names
        self.types = [validate(typ) for typ in types]

        self._name_locs = dict((v, i) for i, v in enumerate(self.names))

        if len(self._name_locs) < len(self.names):
            raise com.IntegrityError('Duplicate column names')

    def __repr__(self):
        space = 2 + max(map(len, self.names))
        return "ibis.Schema {{{}\n}}".format(
            util.indent(
                ''.join(
                    '\n{}{}'.format(name.ljust(space), str(type))
                    for name, type in zip(self.names, self.types)
                ),
                2
            )
        )

    def __hash__(self):
        return hash((type(self), tuple(self.names), tuple(self.types)))

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name):
        return name in self._name_locs

    def __getitem__(self, name):
        return self.types[self._name_locs[name]]

    def __getstate__(self):
        return {
            slot: getattr(self, slot) for slot in self.__class__.__slots__
        }

    def __setstate__(self, instance_dict):
        for key, value in instance_dict.items():
            setattr(self, key, value)

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

        names, types = zip(*values) if values else ([], [])
        return Schema(names, types)

    @classmethod
    def from_dict(cls, dictionary):
        return Schema(*zip(*dictionary.items()))

    def equals(self, other, cache=None):
        return self.names == other.names and self.types == other.types

    def __eq__(self, other):
        return self.equals(other)

    def append(self, schema):
        return Schema(self.names + schema.names, self.types + schema.types)

    def items(self):
        return zip(self.names, self.types)

    def name_at_position(self, i):
        """
        """
        upper = len(self.names) - 1
        if not 0 <= i <= upper:
            raise ValueError(
                'Column index must be between 0 and {:d}, inclusive'.format(
                    upper
                )
            )
        return self.names[i]


class HasSchema(object):

    """
    Base class representing a structured dataset with a well-defined
    schema.

    Base implementation is for tables that do not reference a particular
    concrete dataset or database table.
    """

    def __init__(self, schema, name=None):
        if not isinstance(schema, Schema):
            raise TypeError(
                'schema argument to HasSchema class must be a Schema instance'
            )
        self.schema = schema
        self.name = name

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, repr(self.schema))

    def has_schema(self):
        return True

    def equals(self, other, cache=None):
        return type(self) == type(other) and self.schema.equals(
            other.schema, cache=cache
        )

    def root_tables(self):
        return [self]


class DataType(object):

    __slots__ = 'nullable',

    def __init__(self, nullable=True):
        self.nullable = nullable

    def __call__(self, nullable=True):
        return self._factory(nullable=nullable)

    def _factory(self, nullable=True):
        return type(self)(nullable=nullable)

    def __eq__(self, other):
        return self.equals(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        custom_parts = tuple(
            getattr(self, slot)
            for slot in toolz.unique(self.__slots__ + ('nullable',))
        )
        return hash((type(self),) + custom_parts)

    def __repr__(self):
        return '{}({})'.format(
            self.name,
            ', '.join(
                '{}={!r}'.format(slot, getattr(self, slot))
                for slot in toolz.unique(self.__slots__ + ('nullable',))
            )
        )

    if PY2:
        def __getstate__(self):
            return {
                slot: getattr(self, slot)
                for slot in toolz.unique(self.__slots__ + ('nullable',))
            }

        def __setstate__(self, instance_dict):
            for key, value in instance_dict.items():
                setattr(self, key, value)

    def __str__(self):
        return self.name.lower()

    @property
    def name(self):
        return type(self).__name__

    def equals(self, other, cache=None):
        if isinstance(other, six.string_types):
            other = validate(other)

        return (
            isinstance(other, type(self)) and
            self.nullable == other.nullable and
            self._equal_part(other, cache=cache)
        )

    def _equal_part(self, other, cache=None):
        return True

    def can_implicit_cast(self, other):
        return castable(other, self)

    def scalar_type(self):
        import ibis.expr.types as ir
        return getattr(ir, '{}Scalar'.format(self.name))

    def array_type(self):
        import ibis.expr.types as ir
        return getattr(ir, '{}Column'.format(self.name))

    def valid_literal(self, value):
        raise NotImplementedError(
            'valid_literal not implemented for datatype {}'.format(
                type(self).__name__
            )
        )


class Any(DataType):

    __slots__ = ()

    def valid_literal(self, value):
        return True


class Primitive(DataType):

    __slots__ = ()

    def __repr__(self):
        name = self.name.lower()
        if not self.nullable:
            return '{}[non-nullable]'.format(name)
        return name


class Null(DataType):

    __slots__ = ()

    def valid_literal(self, value):
        return value is None or value is ibis.null


class Variadic(DataType):

    __slots__ = ()


class Boolean(Primitive):

    __slots__ = ()

    def valid_literal(self, value):
        return isinstance(value, bool) or (
            isinstance(value, six.integer_types + (np.integer,)) and
            (value == 0 or value == 1)
        )


Bounds = namedtuple('Bounds', ('lower', 'upper'))


class Integer(Primitive):

    __slots__ = ()

    @property
    def bounds(self):
        exp = self._nbytes * 8 - 1
        lower = -1 << exp
        return Bounds(lower=lower, upper=~lower)

    def valid_literal(self, value):
        lower, upper = self.bounds
        return isinstance(
            value, six.integer_types + (np.integer,)
        ) and lower <= value <= upper


class String(Variadic):
    """A type representing a string.

    Notes
    -----
    Because of differences in the way different backends handle strings, we
    cannot assume that strings are UTF-8 encoded.
    """

    __slots__ = ()

    def valid_literal(self, value):
        return isinstance(value, six.string_types)


class Binary(Variadic):
    """A type representing a blob of bytes.

    Notes
    -----
    Some databases treat strings and blobs of equally, and some do not. For
    example, Impala doesn't make a distinction between string and binary types
    but PostgreSQL has a TEXT type and a BYTEA type which are distinct types
    that behave differently.
    """

    def valid_literal(self, value):
        return isinstance(value, six.binary_type)


class Date(Primitive):

    __slots__ = ()

    def valid_literal(self, value):
        return isinstance(value, six.string_types + (datetime.date,))


class Time(Primitive):

    __slots__ = ()

    def valid_literal(self, value):
        return isinstance(value, six.string_types + (datetime.time,))


def parametric(cls):
    type_name = cls.__name__
    array_type_name = '{}Column'.format(type_name)
    scalar_type_name = '{}Scalar'.format(type_name)

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
class Timestamp(Primitive):

    __slots__ = 'timezone',

    def __init__(self, timezone=None, nullable=True):
        super(Timestamp, self).__init__(nullable=nullable)
        self.timezone = timezone

    def _equal_part(self, other, cache=None):
        return self.timezone == other.timezone

    def __call__(self, timezone=None, nullable=True):
        return type(self)(timezone=timezone, nullable=nullable)

    def __str__(self):
        timezone = self.timezone
        typename = self.name.lower()
        if timezone is None:
            return typename
        return '{}({!r})'.format(typename, timezone)

    def __repr__(self):
        return DataType.__repr__(self)

    def valid_literal(self, value):
        return isinstance(value, six.string_types + (datetime.datetime,))


class SignedInteger(Integer):
    pass


class UnsignedInteger(Integer):

    @property
    def bounds(self):
        exp = self._nbytes * 8 - 1
        upper = 1 << exp
        return Bounds(lower=0, upper=upper)

    def valid_literal(self, value):
        lower, upper = self.bounds
        return isinstance(
            value, six.integer_types + (np.integer,)
        ) and lower <= value <= upper


class Floating(Primitive):

    __slots__ = ()

    def valid_literal(self, value):
        valid_floating_types = (
            builtins.float, np.floating, np.integer
        ) + six.integer_types
        return isinstance(value, valid_floating_types)


class Int8(SignedInteger):

    __slots__ = ()

    _nbytes = 1


class Int16(SignedInteger):

    __slots__ = ()

    _nbytes = 2


class Int32(SignedInteger):

    __slots__ = ()

    _nbytes = 4


class Int64(SignedInteger):

    __slots__ = ()

    _nbytes = 8


class UInt8(UnsignedInteger):

    _nbytes = 1


class UInt16(UnsignedInteger):

    _nbytes = 2


class UInt32(UnsignedInteger):

    _nbytes = 4


class UInt64(UnsignedInteger):

    _nbytes = 8


class Halffloat(Floating):

    _nbytes = 2


class Float(Floating):

    __slots__ = ()

    _nbytes = 4


class Double(Floating):

    __slots__ = ()

    _nbytes = 8


@parametric
class Decimal(DataType):

    __slots__ = 'precision', 'scale'

    def __init__(self, precision, scale, nullable=True):
        super(Decimal, self).__init__(nullable=nullable)
        self.precision = precision
        self.scale = scale

    def __str__(self):
        return '{}({:d}, {:d})'.format(
            self.name.lower(),
            self.precision,
            self.scale,
        )

    def _equal_part(self, other, cache=None):
        return self.precision == other.precision and self.scale == other.scale


assert hasattr(Decimal, '__hash__')


@parametric
class Interval(DataType):

    __slots__ = 'value_type', 'unit'

    _units = dict(
        Y='year',
        Q='quarter',
        M='month',
        w='week',
        d='day',
        h='hour',
        m='minute',
        s='second',
        ms='millisecond',
        us='microsecond',
        ns='nanosecond'
    )

    def __init__(self, unit='s', value_type=None, nullable=True):
        super(Interval, self).__init__(nullable=nullable)
        if unit not in self._units:
            raise ValueError('Unsupported interval unit `{}`'.format(unit))

        if value_type is None:
            value_type = int32
        else:
            value_type = validate(value_type)

        if not isinstance(value_type, Integer):
            raise TypeError("Interval's inner type must be an Integer subtype")

        self.unit = unit
        self.value_type = value_type

    @property
    def resolution(self):
        """Unit's name"""
        return self._units[self.unit]

    def __str__(self):
        unit = self.unit
        typename = self.name.lower()
        value_type_name = self.value_type.name.lower()
        return '{}<{}>(unit={!r})'.format(typename, value_type_name, unit)

    def _equal_part(self, other, cache=None):
        return (self.unit == other.unit and
                self.value_type.equals(other.value_type, cache=cache))

    def valid_literal(self, value):
        return isinstance(value, six.integer_types + (datetime.timedelta,))


@parametric
class Category(DataType):

    __slots__ = 'cardinality',

    def __init__(self, cardinality=None, nullable=True):
        super(Category, self).__init__(nullable=nullable)
        self.cardinality = cardinality

    def __repr__(self):
        if self.cardinality is not None:
            cardinality = self.cardinality
        else:
            cardinality = 'unknown'
        return '{}(cardinality={!r})'.format(self.name, cardinality)

    def _equal_part(self, other, cache=None):
        return (
            self.cardinality == other.cardinality and
            self.nullable == other.nullable
        )

    def to_integer_type(self):
        # TODO: this should be removed I guess
        return int_class(self.cardinality)


@parametric
class Struct(DataType):

    __slots__ = 'pairs',

    def __init__(self, names, types, nullable=True):
        """Construct a ``Struct`` type from a `names` and `types`.

        Parameters
        ----------
        names : Sequence[str]
            Sequence of strings indicating the name of each field in the
            struct.
        types : Sequence[Union[str, DataType]]
            Sequence of strings or :class:`~ibis.expr.datatypes.DataType`
            instances, one for each field
        nullable : bool, optional
            Whether the struct can be null
        """
        if len(names) != len(types):
            raise ValueError('names and types must have the same length')

        super(Struct, self).__init__(nullable=nullable)
        self.pairs = OrderedDict(zip(names, types))

    @classmethod
    def from_tuples(self, pairs):
        return Struct(*map(list, zip(*pairs)))

    @property
    def names(self):
        return self.pairs.keys()

    @property
    def types(self):
        return self.pairs.values()

    def __getitem__(self, key):
        return self.pairs[key]

    def __hash__(self):
        return hash((
            type(self), tuple(self.names), tuple(self.types), self.nullable
        ))

    def __repr__(self):
        return '{}({}, nullable={})'.format(
            self.name, list(self.pairs.items()), self.nullable
        )

    def __str__(self):
        return '{}<{}>'.format(
            self.name.lower(),
            ', '.join(itertools.starmap('{}: {}'.format, self.pairs.items()))
        )

    def _equal_part(self, other, cache=None):
        return self.names == other.names and (
            left.equals(right, cache=cache)
            for left, right in zip(self.types, other.types)
        )

    def valid_literal(self, value):
        """Return whether the type of `value` is a Python literal type
        that can be represented by an ibis ``Struct`` type.

        Parameters
        ----------
        value : object
            Any Python object

        Returns
        -------
        is_valid : bool
            Whether `value` can be used to represent an ibis ``Struct``.
        """
        return isinstance(value, OrderedDict)


@parametric
class Array(Variadic):

    __slots__ = 'value_type',

    def __init__(self, value_type, nullable=True):
        super(Array, self).__init__(nullable=nullable)
        self.value_type = validate(value_type)

    def __str__(self):
        return '{}<{}>'.format(self.name.lower(), self.value_type)

    def _equal_part(self, other, cache=None):
        return self.value_type.equals(other.value_type, cache=cache)

    def valid_literal(self, value):
        return isinstance(value, list)


@parametric
class Enum(DataType):

    __slots__ = 'rep_type', 'value_type'

    def __init__(self, rep_type, value_type, nullable=True):
        super(Enum, self).__init__(nullable=nullable)
        self.rep_type = validate(rep_type)
        self.value_type = validate(value_type)

    def _equal_part(self, other, cache=None):
        return (
            self.rep_type.equals(other.rep_type, cache=cache) and
            self.value_type.equals(other.value_type, cache=cache)
        )


@parametric
class Map(Variadic):

    __slots__ = 'key_type', 'value_type'

    def __init__(self, key_type, value_type, nullable=True):
        super(Map, self).__init__(nullable=nullable)
        self.key_type = validate(key_type)
        self.value_type = validate(value_type)

    def __str__(self):
        return '{}<{}, {}>'.format(
            self.name.lower(),
            self.key_type,
            self.value_type,
        )

    def _equal_part(self, other, cache=None):
        return (
            self.key_type.equals(other.key_type, cache=cache) and
            self.value_type.equals(other.value_type, cache=cache)
        )

    def valid_literal(self, value):
        return isinstance(value, dict)


# ---------------------------------------------------------------------

_builtin_float = float  # TODO: resolve name conflict

any = Any()
null = Null()
boolean = Boolean()
int_ = Integer()
int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
uint_ = UnsignedInteger()
uint8 = UInt8()
uint16 = UInt16()
uint32 = UInt32()
uint64 = UInt64()
float = Float()
halffloat = Halffloat()
float16 = Halffloat()
float32 = Float()
float64 = Double()
double = Double()
string = String()
binary = Binary()
date = Date()
time = Time()
timestamp = Timestamp()
interval = Interval()


_primitive_types = {
    'any': any,
    'null': null,
    'boolean': boolean,
    'int8': int8,
    'int16': int16,
    'int32': int32,
    'int64': int64,
    'uint8': uint8,
    'uint16': uint16,
    'uint32': uint32,
    'uint64': uint64,
    'float': float,
    'halffloat': float16,
    'double': double,
    'string': string,
    'binary': binary,
    'date': date,
    'time': time,
    'timestamp': timestamp,
    'interval': interval
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
    STRARG = 17
    TIMESTAMP = 18
    TIME = 19
    INTERVAL = 20

    @staticmethod
    def name(value):
        return _token_names[value]


_token_names = dict(
    (getattr(Tokens, n), n)
    for n in dir(Tokens) if n.isalpha() and n.isupper()
)


Token = namedtuple('Token', ('type', 'value'))


# Adapted from tokenize.String
_STRING_REGEX = """('[^\n'\\\\]*(?:\\\\.[^\n'\\\\]*)*'|"[^\n"\\\\"]*(?:\\\\.[^\n"\\\\]*)*")"""  # noqa: E501


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
        if token not in {'any', 'null', 'timestamp', 'time', 'interval'}
    ] + [
        # timestamp
        (
            r'(?P<TIMESTAMP>timestamp)',
            lambda token: Token(Tokens.TIMESTAMP, token),
        ),
    ] + [
        # interval - should remove?
        (
            r'(?P<INTERVAL>interval)',
            lambda token: Token(Tokens.INTERVAL, token),
        ),
    ] + [
        # time
        (
            r'(?P<TIME>time)',
            lambda token: Token(Tokens.TIME, token),
        ),
    ] + [
        # decimal + complex types
        (
            '(?P<{}>{})'.format(token.upper(), token),
            lambda token, toktype=toktype: Token(toktype, token)
        ) for token, toktype in zip(
            (
                'decimal',
                'varchar',
                'char',
                'array',
                'map',
                'struct',
                'interval'
            ), (
                Tokens.DECIMAL,
                Tokens.VARCHAR,
                Tokens.CHAR,
                Tokens.ARRAY,
                Tokens.MAP,
                Tokens.STRUCT,
                Tokens.INTERVAL
            ),
        )
    ] + [
        # integers, for decimal spec
        (r'(?P<INTEGER>\d+)', lambda token: Token(Tokens.INTEGER, int(token))),

        # struct fields
        (
            r'(?P<FIELD>[a-zA-Z_][a-zA-Z_0-9]*)',
            lambda token: Token(Tokens.FIELD, token)
        ),
        # timezones
        ('(?P<COMMA>,)', lambda token: Token(Tokens.COMMA, token)),
        ('(?P<COLON>:)', lambda token: Token(Tokens.COLON, token)),
        (r'(?P<LPAREN>\()', lambda token: Token(Tokens.LPAREN, token)),
        (r'(?P<RPAREN>\))', lambda token: Token(Tokens.RPAREN, token)),
        ('(?P<LBRACKET><)', lambda token: Token(Tokens.LBRACKET, token)),
        ('(?P<RBRACKET>>)', lambda token: Token(Tokens.RBRACKET, token)),
        (r'(?P<WHITESPACE>\s+)', None),
        (
            '(?P<STRARG>{})'.format(_STRING_REGEX),
            lambda token: Token(Tokens.STRARG, token),
        ),
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

    __slots__ = 'text', 'tokens', 'tok', 'nexttok'

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
            raise SyntaxError('Expected {} after {!r} in {!r}'.format(
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
                'Found additional tokens {}'.format(additional_tokens)
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
                  | "uint8"
                  | "uint16"
                  | "uint32"
                  | "uint64"
                  | "halffloat"
                  | "float"
                  | "double"
                  | "float16"
                  | "float32"
                  | "float64"
                  | "string"
                  | "time"

        timestamp : "timestamp"
                  | "timestamp" "(" timezone ")"

        interval : "interval"
                 | "interval" "(" unit ")"
                 | "interval" "<" type ">" "(" unit ")"

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

        elif self._accept(Tokens.TIMESTAMP):
            if self._accept(Tokens.LPAREN):
                self._expect(Tokens.STRARG)
                timezone = self.tok.value[1:-1]  # remove surrounding quotes
                self._expect(Tokens.RPAREN)
                return Timestamp(timezone=timezone)
            return timestamp

        elif self._accept(Tokens.TIME):
            return Time()

        elif self._accept(Tokens.INTERVAL):
            if self._accept(Tokens.LBRACKET):
                self._expect(Tokens.PRIMITIVE)
                value_type = self.tok.value
                self._expect(Tokens.RBRACKET)
            else:
                value_type = int32

            if self._accept(Tokens.LPAREN):
                self._expect(Tokens.STRARG)
                unit = self.tok.value[1:-1]  # remove surrounding quotes
                self._expect(Tokens.RPAREN)
            else:
                unit = 's'

            return Interval(unit, value_type)

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
            raise SyntaxError('Type cannot be parsed: {}'.format(self.text))


def validate(t):
    if isinstance(t, DataType):
        return t
    elif isinstance(t, six.string_types):
        return TypeParser(t).parse()
    raise TypeError('Value {!r} is not a valid type or string'.format(t))


validate_type = validate


def array_type(t):
    # compatibility
    return validate(t).array_type()


def scalar_type(t):
    # compatibility
    return validate(t).scalar_type()


_SCALAR_TYPE_PRECEDENCE = {
    'timestamp': 11,
    'double': 10,
    'float': 9,
    'decimal': 8,
    'int64': 7,
    'int32': 6,
    'int16': 5,
    'int8': 4,
    'boolean': 3,
    'string': 2,
    'binary': 1,
    'null': 0,
}


def higher_precedence(left, right):
    left_name = left.name.lower()
    right_name = right.name.lower()

    if (left_name in _SCALAR_TYPE_PRECEDENCE and
            right_name in _SCALAR_TYPE_PRECEDENCE):
        left_prec = _SCALAR_TYPE_PRECEDENCE[left_name]
        right_prec = _SCALAR_TYPE_PRECEDENCE[right_name]
        _, highest_type = max(
            ((left_prec, left), (right_prec, right)),
            key=toolz.first
        )
        return highest_type

    # TODO(phillipc): Ensure that left and right are API compatible

    if isinstance(left, Array):
        return Array(higher_precedence(left.value_type, right.value_type))

    if isinstance(left, Map):
        return Map(
            higher_precedence(left.key_type, right.key_type),
            higher_precedence(left.value_type, right.value_type)
        )

    if isinstance(left, Struct):
        if left.names != right.names:
            raise TypeError('Struct names are not equal')
        return Struct(
            left.names,
            list(map(higher_precedence, left.types, right.types))
        )
    raise TypeError(
        'Cannot compute precedence for {} and {} types'.format(left, right)
    )


def highest_precedence_dtype(dtypes):
    # Return the highest precedence type from the passed expressions. Also
    # verifies that there are valid implicit casts between any of the types and
    # the selected highest precedence type
    if not dtypes:
        raise ValueError('Must pass at least one expression')

    highest_dtype = functools.reduce(higher_precedence, set(dtypes))

    for dtype in dtypes:
        if not castable(dtype, highest_dtype):

            raise TypeError(
                'Datatype {0} cannot be implicitly casted to {1}'
                .format(dtype, highest_dtype)
            )

    return highest_dtype


def int_class(value, allow_overflow=False):
    for dtype in [int8, int16, int32, int64]:
        if dtype.bounds.lower <= value <= dtype.bounds.upper:
            return dtype

    if not allow_overflow:
        raise OverflowError(value)

    return int64


infer = Dispatcher('infer')


@infer.register(object)
def infer_dtype_default(value):
    raise com.InputTypeError(value)


@infer.register(OrderedDict)
def infer_struct(value):
    if not value:
        raise TypeError('Empty struct type not supported')
    return Struct(
        list(value.keys()),
        list(map(infer, value.values()))
    )


@infer.register(dict)
def infer_map(value):
    if not value:
        return Map(null, null)
    return Map(
        highest_precedence_dtype(list(map(infer, value.keys()))),
        highest_precedence_dtype(list(map(infer, value.values()))),
    )


@infer.register(list)
def infer_array(value):
    if not value:
        return Array(null)
    return Array(highest_precedence_dtype(list(map(infer, value))))


# TODO: infer ndarray, infer series


@infer.register(datetime.time)
def infer_time(value):
    return time


@infer.register(datetime.date)
def infer_date(value):
    return date


@infer.register(datetime.datetime)
def infer_timestamp(value):
    return timestamp


# TODO: infer pd.Timedelta, Datetime

@infer.register(datetime.timedelta)
def infer_interval(value):
    return interval


@infer.register(six.string_types)
def infer_string(value):
    return string


@infer.register(_builtin_float)
def infer_floating(value):
    return double


@infer.register(six.integer_types + (np.integer,))
def infer_integer(value):
    return int_class(value)


@infer.register(bool)
def infer_boolean(value):
    return boolean


@infer.register((type(None), Null))
def infer_null(value):
    return null


#  isinstance(a, np.generic) -> if true then map based on a.dtype

# TODO:
# multipledispatch infer_dtype - ala pandas api
# multipledispatch infer_schema - list, dict, ordereddict, dataframe
# to_pandas


castable = Dispatcher('castable')


@castable.register(DataType, DataType)
def can_cast_subtype(source, target):
    return isinstance(target, type(source))


@castable.register(Any, DataType)
def can_cast_any(source, target):
    return True


@castable.register(Null, DataType)
def can_cast_null(source, target):
    return target.nullable


@castable.register(Integer, Integer)
def can_cast_to_generic_integer(source, target):
    return True


@castable.register(UnsignedInteger, UnsignedInteger)
def can_cast_unsigned_integer(source, target):
    return target._nbytes >= source._nbytes


@castable.register(SignedInteger, SignedInteger)
def can_cast_signed_integers(source, target):
    return target._nbytes >= source._nbytes


@castable.register(Floating, Floating)
def can_cast_floats(source, target):
    return target._nbytes >= source._nbytes


@castable.register(Integer, (Floating, Decimal))
def can_upcast_integers(source, target):
    return True


@castable.register(Floating, Decimal)
def can_upcast_floats(source, target):
    return True


@castable.register(Interval, Interval)
def can_cast_intervals(source, target):
    return castable(source.value_type, target.value_type)


# @castable.register(Array, Array)
# def can_cast_arrays(source, target):
#     return (source.equals(target) or
#             source.equals(Array(null)) or
#             source.equals(Array(any)))


# @castable.register(Map, Map)
# def can_cast_maps(source, target):
#     print('EEEEEEEEEEEEEEEEEEEEEE')
#     print(source, target)
#     e = (source.equals(target) or
#             source.equals(Map(null, null)) or
#             source.equals(Map(any, any)))
#     print(e)
#     return e


# TODO cast category


infer_schema = Dispatcher('infer_schema')
