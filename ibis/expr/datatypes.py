import builtins
import collections
import datetime
import functools
import itertools
import numbers
import re
import typing

from typing import (
    Any as GenericAny,
    Callable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set as GenericSet,
    Tuple,
    TypeVar,
    Union,
)

import pandas as pd

import toolz
from multipledispatch import Dispatcher

import ibis.common as com
import ibis.expr.types as ir

GenericDataType = TypeVar('GenericDataType', bound='DataType')
EqualityCache = Mapping[Tuple[GenericDataType, GenericDataType], bool]


class DataType:
    __slots__ = 'nullable',

    def __init__(self, nullable: bool = True) -> None:
        self.nullable = nullable

    def __call__(self, nullable: bool = True) -> 'DataType':
        if nullable is not True and nullable is not False:
            raise TypeError(
                "__call__ only accepts the 'nullable' argument. "
                "Please construct a new instance of the type to change the "
                "values of the attributes."
            )
        return self._factory(nullable=nullable)

    def _factory(self, nullable: bool = True) -> 'DataType':
        slots = {
            slot: getattr(self, slot) for slot in self.__slots__
            if slot != 'nullable'
        }
        return type(self)(nullable=nullable, **slots)

    def __eq__(self, other) -> bool:
        return self.equals(other)

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        custom_parts = tuple(
            getattr(self, slot)
            for slot in toolz.unique(self.__slots__ + ('nullable',))
        )
        return hash((type(self),) + custom_parts)

    def __repr__(self) -> str:
        return '{}({})'.format(
            self.name,
            ', '.join(
                '{}={!r}'.format(slot, getattr(self, slot))
                for slot in toolz.unique(self.__slots__ + ('nullable',))
            )
        )

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def name(self) -> str:
        return type(self).__name__

    def equals(
        self,
        other: 'DataType',
        cache: Optional[Mapping[GenericAny, bool]] = None
    ) -> bool:
        if isinstance(other, str):
            raise TypeError(
                'Comparing datatypes to strings is not allowed. Convert '
                '{!r} to the equivalent DataType instance.'.format(other)
            )
        return (
            isinstance(other, type(self)) and
            self.nullable == other.nullable and
            self.__slots__ == other.__slots__ and
            all(getattr(self, slot) == getattr(other, slot)
                for slot in self.__slots__)
        )

    def castable(self, target, **kwargs):
        return castable(self, target, **kwargs)

    def cast(self, target, **kwargs):
        return cast(self, target, **kwargs)

    def scalar_type(self):
        return functools.partial(self.scalar, dtype=self)

    def column_type(self):
        return functools.partial(self.column, dtype=self)


class Any(DataType):
    __slots__ = ()


class Primitive(DataType):
    __slots__ = ()

    def __repr__(self) -> str:
        name = self.name.lower()
        if not self.nullable:
            return '{}[non-nullable]'.format(name)
        return name


class Null(DataType):
    scalar = ir.NullScalar
    column = ir.NullColumn

    __slots__ = ()


class Variadic(DataType):
    __slots__ = ()


class Boolean(Primitive):
    scalar = ir.BooleanScalar
    column = ir.BooleanColumn

    __slots__ = ()


IntegerBounds = NamedTuple('IntegerBounds', [('lower', int), ('upper', int)])


class Integer(Primitive):
    scalar = ir.IntegerScalar
    column = ir.IntegerColumn

    __slots__ = ()

    @property
    def nbytes(self) -> int:
        raise TypeError(
            'Cannot computer the number of bytes of an abstract integer'
        )

    @property
    def bounds(self) -> IntegerBounds:
        exp = self.nbytes * 8 - 1
        lower = -1 << exp
        return IntegerBounds(lower=lower, upper=~lower)


class String(Variadic):
    """A type representing a string.

    Notes
    -----
    Because of differences in the way different backends handle strings, we
    cannot assume that strings are UTF-8 encoded.
    """
    scalar = ir.StringScalar
    column = ir.StringColumn

    __slots__ = ()


class Binary(Variadic):
    """A type representing a blob of bytes.

    Notes
    -----
    Some databases treat strings and blobs of equally, and some do not. For
    example, Impala doesn't make a distinction between string and binary types
    but PostgreSQL has a TEXT type and a BYTEA type which are distinct types
    that behave differently.
    """
    scalar = ir.BinaryScalar
    column = ir.BinaryColumn

    __slots__ = ()


class Date(Primitive):
    scalar = ir.DateScalar
    column = ir.DateColumn

    __slots__ = ()


class Time(Primitive):
    scalar = ir.TimeScalar
    column = ir.TimeColumn

    __slots__ = ()


class Timestamp(Primitive):
    scalar = ir.TimestampScalar
    column = ir.TimestampColumn

    __slots__ = 'timezone',

    def __init__(
        self,
        timezone: Optional[str] = None,
        nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.timezone = timezone

    def __str__(self) -> str:
        timezone = self.timezone
        typename = self.name.lower()
        if timezone is None:
            return typename
        return '{}({!r})'.format(typename, timezone)

    def __repr__(self) -> str:
        # Can't use super here because the parent method doesn't print the
        # timezone
        return DataType.__repr__(self)


class SignedInteger(Integer):
    @property
    def largest(self) -> 'Int64':
        return int64


class UnsignedInteger(Integer):
    @property
    def largest(self) -> 'UInt64':
        return uint64

    @property
    def bounds(self) -> IntegerBounds:
        exp = self.nbytes * 8 - 1
        upper = 1 << exp
        return IntegerBounds(lower=0, upper=upper)


class Floating(Primitive):
    scalar = ir.FloatingScalar
    column = ir.FloatingColumn

    __slots__ = ()

    @property
    def largest(self) -> 'Float64':
        return float64

    @property
    def nbytes(self) -> int:
        raise TypeError(
            'Cannot compute the number of bytes of an abstract floating point '
            'type'
        )


class Int8(SignedInteger):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 1


class Int16(SignedInteger):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 2


class Int32(SignedInteger):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 4


class Int64(SignedInteger):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 8


class UInt8(UnsignedInteger):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 1


class UInt16(UnsignedInteger):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 2


class UInt32(UnsignedInteger):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 4


class UInt64(UnsignedInteger):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 8


class Halffloat(Floating):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 2


class Float(Floating):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 4


class Double(Floating):
    __slots__ = ()

    @property
    def nbytes(self) -> int:
        return 8


Float16 = Halffloat
Float32 = Float
Float64 = Double


class Decimal(DataType):
    scalar = ir.DecimalScalar
    column = ir.DecimalColumn

    __slots__ = 'precision', 'scale'

    def __init__(
        self,
        precision: int,
        scale: int,
        nullable: bool = True
    ) -> None:
        if not isinstance(precision, numbers.Integral):
            raise TypeError('Decimal type precision must be an integer')
        if not isinstance(scale, numbers.Integral):
            raise TypeError('Decimal type scale must be an integer')
        if precision < 0:
            raise ValueError('Decimal type precision cannot be negative')
        if not precision:
            raise ValueError('Decimal type precision cannot be zero')
        if scale < 0:
            raise ValueError('Decimal type scale cannot be negative')
        if precision < scale:
            raise ValueError(
                'Decimal type precision must be greater than or equal to '
                'scale. Got precision={:d} and scale={:d}'.format(
                    precision, scale
                )
            )

        super().__init__(nullable=nullable)
        self.precision = precision  # type: int
        self.scale = scale  # type: int

    def __str__(self) -> str:
        return '{}({:d}, {:d})'.format(
            self.name.lower(),
            self.precision,
            self.scale,
        )

    @property
    def largest(self) -> 'Decimal':
        return Decimal(38, self.scale)


class Interval(DataType):
    scalar = ir.IntervalScalar
    column = ir.IntervalColumn

    __slots__ = 'value_type', 'unit'

    # based on numpy's units
    _units = dict(
        Y='year',
        Q='quarter',
        M='month',
        W='week',
        D='day',
        h='hour',
        m='minute',
        s='second',
        ms='millisecond',
        us='microsecond',
        ns='nanosecond'
    )

    def __init__(
        self,
        unit: str = 's',
        value_type: Integer = None,
        nullable: bool = True,
    ) -> None:
        super().__init__(nullable=nullable)
        if unit not in self._units:
            raise ValueError('Unsupported interval unit `{}`'.format(unit))

        if value_type is None:
            value_type = int32
        else:
            value_type = dtype(value_type)

        if not isinstance(value_type, Integer):
            raise TypeError("Interval's inner type must be an Integer subtype")

        self.unit = unit
        self.value_type = value_type

    @property
    def bounds(self):
        return self.value_type.bounds

    @property
    def resolution(self):
        """Unit's name"""
        return self._units[self.unit]

    def __str__(self):
        unit = self.unit
        typename = self.name.lower()
        value_type_name = self.value_type.name.lower()
        return '{}<{}>(unit={!r})'.format(typename, value_type_name, unit)


class Category(DataType):
    scalar = ir.CategoryScalar
    column = ir.CategoryColumn

    __slots__ = 'cardinality',

    def __init__(self, cardinality=None, nullable=True):
        super().__init__(nullable=nullable)
        self.cardinality = cardinality

    def __repr__(self):
        if self.cardinality is not None:
            cardinality = self.cardinality
        else:
            cardinality = 'unknown'
        return '{}(cardinality={!r})'.format(self.name, cardinality)

    def to_integer_type(self):
        # TODO: this should be removed I guess
        if self.cardinality is None:
            return int64
        else:
            return infer(self.cardinality)


class Struct(DataType):
    scalar = ir.StructScalar
    column = ir.StructColumn

    __slots__ = 'names', 'types'

    def __init__(
        self,
        names: Sequence[str],
        types: Sequence[DataType],
        nullable: bool = True
    ) -> None:
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

        super().__init__(nullable=nullable)
        self.names = names
        self.types = types

    @classmethod
    def from_tuples(
        self,
        pairs: Sequence[Tuple[str, Union[str, GenericDataType]]],
        nullable: bool = True,
    ) -> 'Struct':
        names, types = zip(*pairs)
        return Struct(list(names), list(map(dtype, types)), nullable=nullable)

    @property
    def pairs(self) -> Mapping:
        return collections.OrderedDict(zip(self.names, self.types))

    def __getitem__(self, key: str) -> DataType:
        return self.pairs[key]

    def __hash__(self) -> int:
        return hash((
            type(self), tuple(self.names), tuple(self.types), self.nullable
        ))

    def __repr__(self) -> str:
        return '{}({}, nullable={})'.format(
            self.name, list(self.pairs.items()), self.nullable
        )

    def __str__(self) -> str:
        return '{}<{}>'.format(
            self.name.lower(),
            ', '.join(itertools.starmap('{}: {}'.format, self.pairs.items()))
        )


class Array(Variadic):
    scalar = ir.ArrayScalar
    column = ir.ArrayColumn

    __slots__ = 'value_type',

    def __init__(
        self,
        value_type: Union[str, DataType],
        nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.value_type = dtype(value_type)

    def __str__(self) -> str:
        return '{}<{}>'.format(self.name.lower(), self.value_type)


class Set(Variadic):
    scalar = ir.SetScalar
    column = ir.SetColumn

    __slots__ = 'value_type',

    def __init__(
        self,
        value_type: Union[str, DataType],
        nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.value_type = dtype(value_type)

    def __str__(self) -> str:
        return '{}<{}>'.format(self.name.lower(), self.value_type)


class Enum(DataType):
    scalar = ir.EnumScalar
    column = ir.EnumColumn

    __slots__ = 'rep_type', 'value_type'

    def __init__(
        self,
        rep_type: DataType,
        value_type: DataType,
        nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.rep_type = dtype(rep_type)
        self.value_type = dtype(value_type)


class Map(Variadic):
    scalar = ir.MapScalar
    column = ir.MapColumn

    __slots__ = 'key_type', 'value_type'

    def __init__(
        self,
        key_type: DataType,
        value_type: DataType,
        nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.key_type = dtype(key_type)
        self.value_type = dtype(value_type)

    def __str__(self) -> str:
        return '{}<{}, {}>'.format(
            self.name.lower(),
            self.key_type,
            self.value_type,
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
uint_ = UnsignedInteger()
uint8 = UInt8()
uint16 = UInt16()
uint32 = UInt32()
uint64 = UInt64()
float = Float()
halffloat = Halffloat()
float16 = Halffloat()
float32 = Float32()
float64 = Float64()
double = Double()
string = String()
binary = Binary()
date = Date()
time = Time()
timestamp = Timestamp()
interval = Interval()
category = Category()


_primitive_types = [
    ('any', any),
    ('null', null),
    ('boolean', boolean),
    ('bool', boolean),
    ('int8', int8),
    ('int16', int16),
    ('int32', int32),
    ('int64', int64),
    ('uint8', uint8),
    ('uint16', uint16),
    ('uint32', uint32),
    ('uint64', uint64),
    ('float16', float16),
    ('float32', float32),
    ('float64', float64),
    ('float', float),
    ('halffloat', float16),
    ('double', double),
    ('string', string),
    ('binary', binary),
    ('date', date),
    ('time', time),
    ('timestamp', timestamp),
    ('interval', interval)
]  # type: List[Tuple[str, DataType]]


class Tokens:
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
    SET = 21

    @staticmethod
    def name(value):
        return _token_names[value]


_token_names = dict(
    (getattr(Tokens, n), n)
    for n in dir(Tokens) if n.isalpha() and n.isupper()
)


Token = collections.namedtuple('Token', ('type', 'value'))


# Adapted from tokenize.String
_STRING_REGEX = """('[^\n'\\\\]*(?:\\\\.[^\n'\\\\]*)*'|"[^\n"\\\\"]*(?:\\\\.[^\n"\\\\]*)*")"""  # noqa: E501


Action = Optional[Callable[[str], Token]]


_TYPE_RULES = collections.OrderedDict(
    [
        # any, null, bool|boolean
        ('(?P<ANY>any)', lambda token: Token(Tokens.ANY, any)),
        ('(?P<NULL>null)', lambda token: Token(Tokens.NULL, null)),
        (
            '(?P<BOOLEAN>bool(?:ean)?)',
            typing.cast(
                Action,
                lambda token: Token(Tokens.PRIMITIVE, boolean),
            ),
        ),
    ] + [
        # primitive types
        (
            '(?P<{}>{})'.format(token.upper(), token),
            typing.cast(
                Action,
                lambda token, value=value: Token(Tokens.PRIMITIVE, value),
            )
        ) for token, value in _primitive_types
        if token not in {
            'any', 'null', 'timestamp', 'time', 'interval', 'boolean'
        }
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
            typing.cast(
                Action,
                lambda token, toktype=toktype: Token(toktype, token)
            )
        ) for token, toktype in zip(
            (
                'decimal',
                'varchar',
                'char',
                'array',
                'set',
                'map',
                'struct',
                'interval',
            ), (
                Tokens.DECIMAL,
                Tokens.VARCHAR,
                Tokens.CHAR,
                Tokens.ARRAY,
                Tokens.SET,
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


def _generate_tokens(pat: GenericAny, text: str) -> Iterator[Token]:
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
    scanner = pat.scanner(text)
    for m in iter(scanner.match, None):
        lastgroup = m.lastgroup
        func = rules[keys[groupindex[lastgroup] - 1]]
        if func is not None:
            yield func(m.group(lastgroup))


class TypeParser:
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

    def __init__(self, text: str):
        self.text = text  # type: str
        self.tokens = _generate_tokens(_TYPE_PATTERN, text)
        self.tok = None  # type: Optional[Token]
        self.nexttok = None  # type: Optional[Token]

    def _advance(self) -> None:
        self.tok, self.nexttok = self.nexttok, next(self.tokens, None)

    def _accept(self, toktype: int) -> bool:
        if self.nexttok is not None and self.nexttok.type == toktype:
            self._advance()
            assert self.tok is not None, \
                'self.tok should not be None when _accept succeeds'
            return True
        return False

    def _expect(self, toktype: int) -> None:
        if not self._accept(toktype):
            raise SyntaxError(
                'Expected {} after {!r} in {!r}'.format(
                    Tokens.name(toktype),
                    getattr(self.tok, 'value', self.tok),
                    self.text,
                )
            )

    def parse(self) -> DataType:
        self._advance()

        # any and null types cannot be nested
        if self._accept(Tokens.ANY) or self._accept(Tokens.NULL):
            assert self.tok is not None, \
                'self.tok was None when parsing ANY or NULL type'
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

    def type(self) -> DataType:
        """
        type : primitive
             | decimal
             | array
             | set
             | map
             | struct

        primitive : "any"
                  | "null"
                  | "bool"
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

        set : "set" "<" type ">"

        map : "map" "<" type "," type ">"

        struct : "struct" "<" field ":" type ("," field ":" type)* ">"

        field : [a-zA-Z_][a-zA-Z_0-9]*
        """
        if self._accept(Tokens.PRIMITIVE):
            assert self.tok is not None
            return self.tok.value

        elif self._accept(Tokens.TIMESTAMP):
            if self._accept(Tokens.LPAREN):
                self._expect(Tokens.STRARG)
                assert self.tok is not None
                timezone = self.tok.value[1:-1]  # remove surrounding quotes
                self._expect(Tokens.RPAREN)
                return Timestamp(timezone=timezone)
            return timestamp

        elif self._accept(Tokens.TIME):
            return Time()

        elif self._accept(Tokens.INTERVAL):
            if self._accept(Tokens.LBRACKET):
                self._expect(Tokens.PRIMITIVE)
                assert self.tok is not None
                value_type = self.tok.value
                self._expect(Tokens.RBRACKET)
            else:
                value_type = int32

            if self._accept(Tokens.LPAREN):
                self._expect(Tokens.STRARG)
                assert self.tok is not None
                unit = self.tok.value[1:-1]  # remove surrounding quotes
                self._expect(Tokens.RPAREN)
            else:
                unit = 's'

            return Interval(unit, value_type)

        elif self._accept(Tokens.DECIMAL):
            if self._accept(Tokens.LPAREN):
                self._expect(Tokens.INTEGER)
                assert self.tok is not None
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

        elif self._accept(Tokens.SET):
            self._expect(Tokens.LBRACKET)

            value_type = self.type()

            self._expect(Tokens.RBRACKET)
            return Set(value_type)

        elif self._accept(Tokens.MAP):
            self._expect(Tokens.LBRACKET)

            self._expect(Tokens.PRIMITIVE)
            assert self.tok is not None
            key_type = self.tok.value

            self._expect(Tokens.COMMA)

            value_type = self.type()

            self._expect(Tokens.RBRACKET)

            return Map(key_type, value_type)

        elif self._accept(Tokens.STRUCT):
            self._expect(Tokens.LBRACKET)

            self._expect(Tokens.FIELD)
            assert self.tok is not None
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


def column_type(t):
    # compatibility
    return dtype(t).column_type()


def scalar_type(t):
    # compatibility
    return dtype(t).scalar_type()


dtype = Dispatcher('dtype')

validate_type = dtype


@dtype.register(object)
def default(value, **kwargs) -> DataType:
    raise com.IbisTypeError('Value {!r} is not a valid datatype'.format(value))


@dtype.register(DataType)
def from_ibis_dtype(value: DataType) -> DataType:
    return value


@dtype.register(str)
def from_string(value: str) -> DataType:
    try:
        return TypeParser(value).parse()
    except SyntaxError:
        raise com.IbisTypeError(
            '{!r} cannot be parsed as a datatype'.format(value)
        )


@dtype.register(list)
def from_list(values: List[GenericAny]) -> Array:
    if not values:
        return Array(null)
    return Array(highest_precedence(map(dtype, values)))


@dtype.register(collections.Set)
def from_set(values: GenericSet) -> Set:
    if not values:
        return Set(null)
    return Set(highest_precedence(map(dtype, values)))


infer = Dispatcher('infer')


def higher_precedence(left: DataType, right: DataType) -> DataType:
    if castable(left, right, upcast=True):
        return right
    elif castable(right, left, upcast=True):
        return left

    raise com.IbisTypeError(
        'Cannot compute precedence for {} and {} types'.format(left, right)
    )


def highest_precedence(dtypes: Iterator[DataType]) -> DataType:
    return functools.reduce(higher_precedence, dtypes)


@infer.register(object)
def infer_dtype_default(value: GenericAny) -> DataType:
    raise com.InputTypeError(value)


@infer.register(collections.OrderedDict)
def infer_struct(value: collections.OrderedDict) -> Struct:
    if not value:
        raise TypeError('Empty struct type not supported')
    return Struct(
        list(value.keys()),
        list(map(infer, value.values()))
    )


@infer.register(collections.abc.Mapping)
def infer_map(value: Mapping) -> Map:
    if not value:
        return Map(null, null)
    return Map(
        highest_precedence(map(infer, value.keys())),
        highest_precedence(map(infer, value.values())),
    )


@infer.register(list)
def infer_list(values: List[GenericAny]) -> Array:
    if not values:
        return Array(null)
    return Array(highest_precedence(map(infer, values)))


@infer.register((set, frozenset))
def infer_set(values: GenericSet) -> Set:
    if not values:
        return Set(null)
    return Set(highest_precedence(map(infer, values)))


@infer.register(datetime.time)
def infer_time(value: datetime.time) -> Time:
    return time


@infer.register(datetime.date)
def infer_date(value: datetime.date) -> Date:
    return date


@infer.register(datetime.datetime)
def infer_timestamp(value: datetime.datetime) -> Timestamp:
    if value.tzinfo:
        return Timestamp(timezone=str(value.tzinfo))
    else:
        return timestamp


@infer.register(datetime.timedelta)
def infer_interval(value: datetime.timedelta) -> Interval:
    return interval


@infer.register(str)
def infer_string(value: str) -> String:
    return string


@infer.register(builtins.float)
def infer_floating(value: builtins.float) -> Double:
    return double


@infer.register(int)
def infer_integer(value: int, allow_overflow: bool = False) -> Integer:
    for dtype in (int8, int16, int32, int64):
        if dtype.bounds.lower <= value <= dtype.bounds.upper:
            return dtype

    if not allow_overflow:
        raise OverflowError(value)

    return int64


@infer.register(bool)
def infer_boolean(value: bool) -> Boolean:
    return boolean


@infer.register((type(None), Null))
def infer_null(value: Optional[Null]) -> Null:
    return null


castable = Dispatcher('castable')


@castable.register(DataType, DataType)
def can_cast_subtype(source: DataType, target: DataType, **kwargs) -> bool:
    return isinstance(target, type(source))


@castable.register(Any, DataType)
@castable.register(DataType, Any)
@castable.register(Any, Any)
@castable.register(Null, Any)
@castable.register(Integer, Category)
@castable.register(Integer, (Floating, Decimal))
@castable.register(Floating, Decimal)
@castable.register((Date, Timestamp), (Date, Timestamp))
def can_cast_any(source, target, **kwargs) -> bool:
    return True


@castable.register(Null, DataType)
def can_cast_null(source, target, **kwargs) -> bool:
    return target.nullable


@castable.register(SignedInteger, UnsignedInteger)
@castable.register(UnsignedInteger, SignedInteger)
def can_cast_to_unsigned(
    source: Integer,
    target: Integer,
    value: Optional[int] = None,
    **kwargs
) -> bool:
    if value is None:
        return False

    bounds = target.bounds
    return bounds.lower <= value <= bounds.upper


@castable.register(SignedInteger, SignedInteger)
@castable.register(UnsignedInteger, UnsignedInteger)
def can_cast_integers(source: Integer, target: Integer, **kwargs) -> bool:
    return target.nbytes >= source.nbytes


@castable.register(Floating, Floating)
def can_cast_floats(
    source: Floating,
    target: Floating,
    upcast: bool = False,
    **kwargs
) -> bool:
    if upcast:
        return target.nbytes >= source.nbytes

    # double -> float must be allowed because
    # float literals are inferred as doubles
    return True


@castable.register(Decimal, Decimal)
def can_cast_decimals(source: Decimal, target: Decimal, **kwargs) -> bool:
    return (target.precision >= source.precision and
            target.scale >= source.scale)


@castable.register(Interval, Interval)
def can_cast_intervals(source: Interval, target: Interval, **kwargs) -> bool:
    return (
        source.unit == target.unit and
        castable(source.value_type, target.value_type)
    )


@castable.register(Integer, Boolean)
def can_cast_integer_to_boolean(
    source: Integer,
    target: Boolean,
    value: Optional[int] = None,
    **kwargs
) -> bool:
    return value is not None and (value == 0 or value == 1)


@castable.register(Integer, Interval)
def can_cast_integer_to_interval(
    source: Interval,
    target: Interval,
    **kwargs
) -> bool:
    return castable(source, target.value_type)


@castable.register(String, (Date, Time, Timestamp))
def can_cast_string_to_temporal(
    source: String,
    target: Union[Date, Time, Timestamp],
    value: Optional[str] = None,
    **kwargs
) -> bool:
    if value is None:
        return False
    try:
        # this is the only pandas import left
        pd.Timestamp(value)
        return True
    except ValueError:
        return False


Collection = TypeVar('Collection', Array, Set)


@castable.register(Array, Array)
@castable.register(Set, Set)
def can_cast_variadic(source: Collection, target: Collection, **kwargs):
    return castable(source.value_type, target.value_type)


# @castable.register(Map, Map)
# def can_cast_maps(source, target):
#     return (source.equals(target) or
#             source.equals(Map(null, null)) or
#             source.equals(Map(any, any)))
# TODO cast category


def cast(source, target, **kwargs) -> DataType:
    """Attempts to implicitly cast from source dtype to target dtype"""
    source, target = dtype(source), dtype(target)

    if not castable(source, target, **kwargs):
        raise com.IbisTypeError('Datatype {} cannot be implicitly '
                                'casted to {}'.format(source, target))
    return target


same_kind = Dispatcher(
    'same_kind',
    doc="""\
Compute whether two :class:`~ibis.expr.datatypes.DataType` instances are the
same kind.

Parameters
----------
a : DataType
b : DataType

Returns
-------
bool
    Whether two :class:`~ibis.expr.datatypes.DataType` instances are the same
    kind.
""")


@same_kind.register(DataType, DataType)
def same_kind_default(a: DataType, b: DataType) -> bool:
    """Return whether `a` is exactly equiavlent to `b`"""
    return a.equals(b)


Numeric = TypeVar('Numeric', Integer, Floating)


@same_kind.register(Integer, Integer)
@same_kind.register(Floating, Floating)
def same_kind_numeric(a: Numeric, b: Numeric) -> bool:
    """Return ``True``."""
    return True


@same_kind.register(DataType, Null)
def same_kind_right_null(a: DataType, _: Null) -> bool:
    """Return whether `a` is nullable."""
    return a.nullable


@same_kind.register(Null, DataType)
def same_kind_left_null(_: Null, b: DataType) -> bool:
    """Return whether `b` is nullable."""
    return b.nullable


@same_kind.register(Null, Null)
def same_kind_both_null(a: Null, b: Null) -> bool:
    """Return ``True``."""
    return True
