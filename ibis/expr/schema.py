import collections
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from multipledispatch import Dispatcher

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.util as util

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


class Schema:

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
        self.types = list(map(dt.dtype, types))

        self._name_locs = {v: i for i, v in enumerate(self.names)}

        if len(self._name_locs) < len(self.names):
            duplicate_names = list(self.names)
            for v in self._name_locs.keys():
                duplicate_names.remove(v)
            raise com.IntegrityError(
                f'Duplicate column name(s): {duplicate_names}'
            )

    def __repr__(self):
        space = 2 + max(map(len, self.names), default=0)
        return "ibis.Schema {{{}\n}}".format(
            util.indent(
                ''.join(
                    f'\n{name.ljust(space)}{str(type)}'
                    for name, type in zip(self.names, self.types)
                ),
                2,
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
        return {slot: getattr(self, slot) for slot in self.__class__.__slots__}

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

        for column, dtype in self.items():
            pandas_dtype = dtype.to_pandas()
            col = df[column]
            col_dtype = col.dtype

            try:
                not_equal = pandas_dtype != col_dtype
            except TypeError:
                # ugh, we can't compare dtypes coming from pandas,
                # assume not equal
                not_equal = True

            if not_equal or isinstance(dtype, dt.String):
                df[column] = convert(col_dtype, dtype, col)

        return df


class HasSchema:

    """
    Base class representing a structured dataset with a well-defined
    schema.

    Base implementation is for tables that do not reference a particular
    concrete dataset or database table.
    """

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.schema)})'

    def has_schema(self):
        return True

    def equals(self, other, cache=None):
        return type(self) == type(other) and self.schema.equals(
            other.schema, cache=cache
        )

    def root_tables(self):
        return [self]

    @property
    def schema(self):
        raise NotImplementedError


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


def coerce_to_tuple(
    data: Union[List, np.ndarray, pd.Series],
    output_type: dt.Struct,
    index: Optional[pd.Index] = None,
) -> Tuple:
    """Coerce the following shapes to a tuple.

    (1) A list
    (2) An np.ndarray
    (3) A Series
    """
    return tuple(data)


def coerce_to_np_array(
    data: Union[List, np.ndarray, pd.Series],
    output_type: dt.Struct,
    index: Optional[pd.Index] = None,
) -> np.ndarray:
    """Coerce the following shapes to an np.ndarray.

    (1) A list
    (2) An np.ndarray
    (3) A Series
    """
    return np.array(data)


def coerce_to_series(
    data: Union[List, np.ndarray, pd.Series],
    output_type: dt.DataType,
    original_index: Optional[pd.Index] = None,
) -> pd.Series:
    """Coerce the following shapes to a Series.

    (1) A list
    (2) An np.ndarray
    (3) A Series

    Note:
    This method does NOT always return a new Series. If a Series is
    passed in, this method will return the original object.

    Parameters
    ----------
    data : pd.Series, a list, or an np.array

    output_type : the type of the output

    original_index : Optional parameter containing the index of the output

    Returns
    -------
    pd.Series
    """
    if isinstance(data, (list, np.ndarray)):
        result = pd.Series(data)
    elif isinstance(data, pd.Series):
        result = data
    else:
        # This case is a non-vector elementwise or analytic UDF that should
        # not be coerced to a Series.
        return data
    if original_index is not None:
        result.index = original_index
    return result


def coerce_to_dataframe(
    data: Any,
    output_type: dt.Struct,
    original_index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Coerce the following shapes to a DataFrame.

    The following shapes are allowed:
    (1) A list/tuple of Series
    (2) A list/tuple np.ndarray
    (3) A list/tuple of scalars
    (4) A Series of list/tuple
    (5) pd.DataFrame

    Note:
    This method does NOT always return a new DataFrame. If a DataFrame is
    passed in, this method will return the original object.

    Parameters
    ----------
    data : pd.DataFrame, a tuple/list of pd.Series, a tuple/list of np.array
    or a pd.Series of tuple/list

    output_type : a Struct containing the names and types of the output

    original_index : Optional parameter containing the index of the output

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> coerce_to_dataframe(pd.DataFrame({'a': [1, 2, 3]}), dt.Struct([('b', 'int32')]))  # noqa: E501
       b
    0  1
    1  2
    2  3
    dtype: int32
    >>> coerce_to_dataframe(pd.Series([[1, 2, 3]]), dt.Struct([('a', 'int32'), ('b', 'int32'), ('c', 'int32')]))  # noqa: E501
       a  b  c
    0  1  2  3
    dtypes: [int32, int32, int32]
    >>> coerce_to_dataframe(pd.Series([range(3), range(3)]), dt.Struct([('a', 'int32'), ('b', 'int32'), ('c', 'int32')]))  # noqa: E501
       a  b  c
    0  0  1  2
    1  0  1  2
    dtypes: [int32, int32, int32]
    >>> coerce_to_dataframe([pd.Series(x) for x in [1, 2, 3]], dt.Struct([('a', 'int32'), ('b', 'int32'), ('c', 'int32')]))  # noqa: E501
       a  b  c
    0  1  2  3
    >>>  coerce_to_dataframe([1, 2, 3], dt.Struct([('a', 'int32'), ('b', 'int32'), ('c', 'int32')]))  # noqa: E501
       a  b  c
    0  1  2  3
    dtypes: [int32, int32, int32]
    """
    if isinstance(data, pd.DataFrame):
        result = data
    elif isinstance(data, pd.Series):
        if not len(data):
            result = data.to_frame()
        else:
            num_cols = len(data.iloc[0])
            series = [data.apply(lambda t: t[i]) for i in range(num_cols)]
            result = pd.concat(series, axis=1)
    elif isinstance(data, (tuple, list, np.ndarray)):
        if isinstance(data[0], pd.Series):
            result = pd.concat(data, axis=1)
        elif isinstance(data[0], np.ndarray):
            result = pd.concat([pd.Series(v) for v in data], axis=1)
        else:
            # Promote scalar to Series
            result = pd.concat([pd.Series([v]) for v in data], axis=1)
    else:
        raise ValueError(f"Cannot coerce to DataFrame: {data}")

    result.columns = output_type.names
    if original_index is not None:
        result.index = original_index
    return result
