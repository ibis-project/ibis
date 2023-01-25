from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping

from multipledispatch import Dispatcher

import ibis.expr.datatypes as dt
from ibis.common.annotations import attribute
from ibis.common.exceptions import IntegrityError
from ibis.common.grounds import Concrete
from ibis.common.validators import frozendict_of, instance_of, validator
from ibis.util import deprecated, indent, warn_deprecated

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


class Schema(Concrete):
    """An object for holding table schema information."""

    fields = frozendict_of(instance_of(str), datatype)
    """A mapping of [`str`][str] to [`DataType`][ibis.expr.datatypes.DataType] objects
    representing the type of each column."""

    @classmethod
    def __create__(cls, names, types=None):
        if types is None:
            return super().__create__(fields=names)
        else:
            warn_deprecated(
                "Schema(names, types)",
                as_of="4.1",
                removed_in="5.0",
                instead=(
                    "construct a Schema using a mapping of names to types instead: "
                    "Schema(dict(zip(names, types)))"
                ),
            )
            return schema(names, types)

    def __reduce__(self):
        return (self.__class__, (self.fields, None))

    def copy(self, fields=None):
        if fields is None:
            fields = self.fields
        return type(self)(fields)

    def __repr__(self) -> str:
        space = 2 + max(map(len, self.names), default=0)
        return "ibis.Schema {{{}\n}}".format(
            indent(
                ''.join(
                    f'\n{name.ljust(space)}{str(type)}'
                    for name, type in self.fields.items()
                ),
                2,
            )
        )

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterable[str]:
        return iter(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    def __getitem__(self, name: str) -> dt.DataType:
        return self.fields[name]

    @attribute.default
    def names(self):
        return tuple(self.fields.keys())

    @attribute.default
    def types(self):
        return tuple(self.fields.values())

    @attribute.default
    def _name_locs(self) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.names)}

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
        >>> assert first.equals(second)
        >>> third = ibis.schema({"a": "array<int>"})
        >>> assert not first.equals(third)
        """
        if not isinstance(other, Schema):
            raise TypeError(
                f"invalid equality comparison between Schema and {type(other)}"
            )
        return self.__cached_equals__(other)

    @deprecated(
        as_of="4.1",
        removed_in="5.0",
        instead="construct a new Schema without the undesired names instead",
    )
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

        delete = frozenset(names_to_delete)
        fields = {k: v for k, v in self.fields.items() if k not in delete}

        return self.__class__(fields)

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
        return cls(dict(values))

    @classmethod
    @deprecated(
        as_of="4.1",
        removed_in="5.0",
        instead="directly construct a Schema instead",
    )
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
        return cls(dictionary)

    def to_pandas(self):
        """Return the equivalent pandas datatypes."""
        from ibis.backends.pandas.client import ibis_schema_to_pandas

        return ibis_schema_to_pandas(self)

    def to_pyarrow(self):
        """Return the equivalent pyarrow schema."""
        from ibis.backends.pyarrow.datatypes import ibis_to_pyarrow_schema

        return ibis_to_pyarrow_schema(self)

    def as_struct(self) -> dt.Struct:
        return dt.Struct(dict(self.items()))

    def __gt__(self, other: Schema) -> bool:
        """Return whether `self` is a strict superset of `other`."""
        return set(self.items()) > set(other.items())

    def __ge__(self, other: Schema) -> bool:
        """Return whether `self` is a superset of or equal to `other`."""
        return set(self.items()) >= set(other.items())

    @deprecated(as_of="4.1", removed_in="5.0", instead="use Schema.merge() instead")
    def append(self, other: Schema) -> Schema:
        return self.merge(other)

    def merge(self, other: Schema) -> Schema:
        """Merge `other` to `self`.

        Raise an `IntegrityError` if there are duplicate column names.

        Parameters
        ----------
        other
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
        >>> first.merge(second)
        ibis.Schema {
          a  int64
          b  string
          c  float64
          d  int16
        }
        """
        if duplicates := self.fields.keys() & other.fields.keys():
            raise IntegrityError(f'Duplicate column name(s): {duplicates}')
        return self.__class__({**self.fields, **other.fields})

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
            raise ValueError(f'Column index must be between 0 and {upper:d}, inclusive')
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

            if not_equal or not dtype.is_primitive():
                new_col = convert(col_dtype, dtype, col)
            else:
                new_col = col
            df[column] = new_col

        # return data with the schema's columns which may be different than the
        # input columns
        df.columns = schema_names
        return df


schema = Dispatcher('schema')
infer = Dispatcher('infer')


@schema.register(Schema)
def identity(s):
    return s


@schema.register(collections.abc.Mapping)
def schema_from_mapping(d):
    return Schema(d)


@schema.register(collections.abc.Iterable)
def schema_from_pairs(lst):
    return Schema.from_tuples(lst)


@schema.register(collections.abc.Iterable, collections.abc.Iterable)
def schema_from_names_types(names, types):
    # validate lengths of names and types are the same
    if len(names) != len(types):
        raise IntegrityError('Schema names and types must have the same length')

    # validate unique field names
    name_locs = {v: i for i, v in enumerate(names)}
    if len(name_locs) < len(names):
        duplicate_names = list(names)
        for v in name_locs:
            duplicate_names.remove(v)
        raise IntegrityError(f'Duplicate column name(s): {duplicate_names}')

    # construct the schema
    fields = dict(zip(names, types))
    return Schema(fields)
