from __future__ import annotations

import abc
import collections
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping, Sequence

from multipledispatch import Dispatcher

from ibis.common.exceptions import IntegrityError
from ibis.common.grounds import Annotable, Comparable
from ibis.common.validators import (
    immutable_property,
    instance_of,
    tuple_of,
    validator,
)
from ibis.expr import datatypes as dt
from ibis.util import UnnamedMarker, deprecated, indent

if TYPE_CHECKING:
    import pandas as pd

convert = Dispatcher(
    'convert',
    doc="""\
Convert `column` to the pandas dtype corresponding to `out_dtype`, where the
dtype of `column` is `in_dtype`.

Parameters
----------
in_dtype : Union[np.dtype, pandas_dtype]
    The dtype of `column`, used for dispatching
out_dtype : ibis.expr.datatypes.DataType
    The requested ibis type of the output
column : pd.Series
    The column to convert

Returns
-------
result : pd.Series
    The converted column
""",
)


@validator
def datatype(arg, **kwargs):
    return dt.dtype(arg)


class Schema(Annotable, Comparable):
    """An object for holding table schema information."""

    __slots__ = ('_name_locs',)

    names: Sequence[str] = tuple_of(instance_of((str, UnnamedMarker)))
    """A sequence of [`str`][str] indicating the name of each column."""
    types: Sequence[dt.DataType] = tuple_of(datatype)
    """A sequence of [DataType][ibis.expr.datatypes.DataType] objects representing type of each column."""  # noqa: E501

    @immutable_property
    def _name_locs(self) -> dict[str, int]:
        # validate unique field names
        name_locs = {v: i for i, v in enumerate(self.names)}
        if len(name_locs) < len(self.names):
            duplicate_names = list(self.names)
            for v in name_locs.keys():
                duplicate_names.remove(v)
            raise IntegrityError(
                f'Duplicate column name(s): {duplicate_names}'
            )
        return name_locs

    def __repr__(self) -> str:
        space = 2 + max(map(len, self.names), default=0)
        return "ibis.Schema {{{}\n}}".format(
            indent(
                ''.join(
                    f'\n{name.ljust(space)}{str(type)}'
                    for name, type in zip(self.names, self.types)
                ),
                2,
            )
        )

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterable[str]:
        return iter(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self._name_locs

    def __getitem__(self, name: str) -> dt.DataType:
        return self.types[self._name_locs[name]]

    def __equals__(self, other: Schema) -> bool:
        return (
            self._hash == other._hash
            and self.names == other.names
            and self.types == other.types
        )

    def equals(self, other: Schema) -> bool:
        """Return whether `other` is equal to `self`.

        Parameters
        ----------
        other
            Schema to compare `self` to.

        Examples
        --------
        >>> import ibis
        >>> first = ibis.schema({"a": "int"})
        >>> second = ibis.schema({"a": "int"})
        >>> first.equals(second)
        True
        >>> third = ibis.schema({"a": "array<int>"})
        >>> first.equals(third)
        False
        """
        if not isinstance(other, Schema):
            raise TypeError(
                "invalid equality comparison between Schema and "
                f"{type(other)}"
            )
        return self.__cached_equals__(other)

    def delete(self, names_to_delete: Iterable[str]) -> Schema:
        """Remove `names_to_delete` names from `self`.

        Parameters
        ----------
        names_to_delete
            Iterable of `str` to remove from the schema.

        Examples
        --------
        >>> import ibis
        >>> sch = ibis.schema({"a": "int", "b": "string"})
        >>> sch.delete({"a"})
        ibis.Schema {
          b  string
        }
        """
        for name in names_to_delete:
            if name not in self:
                raise KeyError(name)

        new_names, new_types = [], []
        for name, type_ in zip(self.names, self.types):
            if name in names_to_delete:
                continue
            new_names.append(name)
            new_types.append(type_)

        return self.__class__(new_names, new_types)

    @classmethod
    def from_tuples(
        cls,
        values: Iterable[tuple[str, str | dt.DataType]],
    ) -> Schema:
        """Construct a `Schema` from an iterable of pairs.

        Parameters
        ----------
        values
            An iterable of pairs of name and type.

        Returns
        -------
        Schema
            A new schema

        Examples
        --------
        >>> import ibis
        >>> ibis.Schema.from_tuples([("a", "int"), ("b", "string")])
        ibis.Schema {
          a  int64
          b  string
        }
        """
        if not isinstance(values, (list, tuple)):
            values = list(values)

        names, types = zip(*values) if values else ([], [])
        return cls(names, types)

    @classmethod
    def from_dict(cls, dictionary: Mapping[str, str | dt.DataType]) -> Schema:
        """Construct a `Schema` from a `Mapping`.

        Parameters
        ----------
        dictionary
            Mapping from which to construct a `Schema` instance.

        Returns
        -------
        Schema
            A new schema

        Examples
        --------
        >>> import ibis
        >>> ibis.Schema.from_dict({"a": "int", "b": "string"})
        ibis.Schema {
          a  int64
          b  string
        }
        """
        names, types = zip(*dictionary.items()) if dictionary else ([], [])
        return cls(names, types)

    def __gt__(self, other: Schema) -> bool:
        """Return whether `self` is a strict superset of `other`."""
        return set(self.items()) > set(other.items())

    def __ge__(self, other: Schema) -> bool:
        """Return whether `self` is a superset of or equal to `other`."""
        return set(self.items()) >= set(other.items())

    def append(self, schema: Schema) -> Schema:
        """Append `schema` to `self`.

        Parameters
        ----------
        schema
            Schema instance to append to `self`.

        Returns
        -------
        Schema
            A new schema appended with `schema`.

        Examples
        --------
        >>> import ibis
        >>> first = ibis.Schema.from_dict({"a": "int", "b": "string"})
        >>> second = ibis.Schema.from_dict({"c": "float", "d": "int16"})
        >>> first.append(second)
        ibis.Schema {
          a  int64
          b  string
          c  float64
          d  int16
        }
        """
        return self.__class__(
            self.names + schema.names, self.types + schema.types
        )

    def items(self) -> Iterator[tuple[str, dt.DataType]]:
        """Return an iterator of pairs of names and types.

        Returns
        -------
        Iterator[tuple[str, dt.DataType]]
            Iterator of schema components

        Examples
        --------
        >>> import ibis
        >>> sch = ibis.Schema.from_dict({"a": "int", "b": "string"})
        >>> list(sch.items())
        [('a', Int64(nullable=True)), ('b', String(nullable=True))]
        """
        return zip(self.names, self.types)

    def name_at_position(self, i: int) -> str:
        """Return the name of a schema column at position `i`.

        Parameters
        ----------
        i
            The position of the column

        Returns
        -------
        str
            The name of the column in the schema at position `i`.

        Examples
        --------
        >>> import ibis
        >>> sch = ibis.Schema.from_dict({"a": "int", "b": "string"})
        >>> sch.name_at_position(0)
        'a'
        >>> sch.name_at_position(1)
        'b'
        """
        upper = len(self.names) - 1
        if not 0 <= i <= upper:
            raise ValueError(
                'Column index must be between 0 and {:d}, inclusive'.format(
                    upper
                )
            )
        return self.names[i]

    def apply_to(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the schema `self` to a pandas `DataFrame`.

        This method mutates the input `DataFrame`.

        Parameters
        ----------
        df
            Input DataFrame

        Returns
        -------
        DataFrame
            Type-converted DataFrame

        Examples
        --------
        Import the necessary modules

        >>> import numpy as np
        >>> import pandas as pd
        >>> import ibis
        >>> import ibis.expr.datatypes as dt

        Construct a DataFrame with string timestamps and an `int8` column that
        we're going to upcast.

        >>> data = dict(
        ...     times=[
        ...         "2022-01-01 12:00:00",
        ...         "2022-01-01 13:00:01",
        ...         "2022-01-01 14:00:02",
        ...     ],
        ...     x=np.array([-1, 0, 1], dtype="int8")
        ... )
        >>> df = pd.DataFrame(data)
        >>> df
                         times  x
        0  2022-01-01 12:00:00 -1
        1  2022-01-01 13:00:01  0
        2  2022-01-01 14:00:02  1
        >>> df.dtypes
        times    object
        x          int8
        dtype: object

        Construct an ibis Schema that we want to cast to.

        >>> sch = ibis.schema({"times": dt.timestamp, "x": "int16"})
        >>> sch
        ibis.Schema {
          times  timestamp
          x      int16
        }

        Apply the schema

        >>> sch.apply_to(df)
                        times  x
        0 2022-01-01 12:00:00 -1
        1 2022-01-01 13:00:01  0
        2 2022-01-01 14:00:02  1
        >>> df.dtypes  # `df` is mutated by the method
        times    datetime64[ns]
        x                 int16
        dtype: object
        """
        schema_names = self.names
        data_columns = df.columns

        assert len(schema_names) == len(
            data_columns
        ), "schema column count does not match input data column count"

        for column, dtype in zip(data_columns, self.types):
            pandas_dtype = dtype.to_pandas()

            col = df[column]
            col_dtype = col.dtype

            try:
                not_equal = pandas_dtype != col_dtype
            except TypeError:
                # ugh, we can't compare dtypes coming from pandas,
                # assume not equal
                not_equal = True

            if not_equal or not isinstance(dtype, dt.Primitive):
                new_col = convert(col_dtype, dtype, col)
            else:
                new_col = col
            df[column] = new_col

        # return data with the schema's columns which may be different than the
        # input columns
        df.columns = schema_names
        return df


class HasSchema(abc.ABC):
    """Mixin representing a structured dataset with a schema."""

    @deprecated(version="4.0", instead="")
    def has_schema(self):
        return True

    def root_tables(self):
        return [self]

    @property
    @abc.abstractmethod
    def schema(self) -> Schema:
        """Return a schema."""


schema = Dispatcher('schema')
infer = Dispatcher('infer')


@schema.register(Schema)
def identity(s):
    return s


@schema.register(collections.abc.Mapping)
def schema_from_mapping(d):
    return Schema.from_dict(d)


@schema.register(collections.abc.Iterable)
def schema_from_pairs(lst):
    return Schema.from_tuples(lst)


@schema.register(collections.abc.Iterable, collections.abc.Iterable)
def schema_from_names_types(names, types):
    return Schema(names, types)
