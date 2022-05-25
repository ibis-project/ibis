from __future__ import annotations

import abc
import collections
from typing import Sequence

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

    __slots__ = ('_name_locs',)

    names: Sequence[str] = tuple_of(instance_of((str, UnnamedMarker)))
    types: Sequence[dt.DataType] = tuple_of(datatype)

    @immutable_property
    def _name_locs(self):
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

    def __repr__(self):
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

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name):
        return name in self._name_locs

    def __getitem__(self, name):
        return self.types[self._name_locs[name]]

    def __equals__(self, other):
        return (
            self._hash == other._hash
            and self.names == other.names
            and self.types == other.types
        )

    def equals(self, other):
        if not isinstance(other, Schema):
            raise TypeError(
                "invalid equality comparison between Schema and "
                f"{type(other)}"
            )
        return self.__cached_equals__(other)

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
        names, types = zip(*dictionary.items()) if dictionary else ([], [])
        return Schema(names, types)

    def __gt__(self, other):
        return set(self.items()) > set(other.items())

    def __ge__(self, other):
        return set(self.items()) >= set(other.items())

    def append(self, schema):
        return Schema(self.names + schema.names, self.types + schema.types)

    def items(self):
        return zip(self.names, self.types)

    def name_at_position(self, i):
        """ """
        upper = len(self.names) - 1
        if not 0 <= i <= upper:
            raise ValueError(
                'Column index must be between 0 and {:d}, inclusive'.format(
                    upper
                )
            )
        return self.names[i]

    def apply_to(self, df):
        """Applies the Ibis schema to a pandas DataFrame

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        df : pandas.DataFrame

        Notes
        -----
        Mutates `df`
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

            if not_equal or isinstance(dtype, (dt.String, dt.Struct)):
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
