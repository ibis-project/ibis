import numpy as np
import pandas as pd
from multipledispatch import Dispatcher

import ibis.common as com
import ibis.util as util
from ibis.expr import datatypes as dt


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
        self.types = list(map(dt.dtype, types))

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


schema = Dispatcher('schema')


@schema.register(Schema)
def identity(s):
    return s


@schema.register(pd.Series)
def schema_from_series(s):
    return Schema.from_tuples(s.iteritems())


@schema.register((tuple, list))
def schema_from_list(lst):
    return Schema.from_tuples(lst)


@schema.register(dict)
def schema_from_dict(d):
    return Schema.from_dict(d)


infer = Dispatcher('infer')


try:
    infer_pandas_dtype = pd.api.types.infer_dtype
except AttributeError:
    infer_pandas_dtype = pd.lib.infer_dtype


@infer.register(pd.DataFrame)
def infer_pandas_schema(df):
    pairs = []
    for column_name, pandas_dtype in df.dtypes.iteritems():
        if pandas_dtype == np.object_:
            pandas_dtype = infer_pandas_dtype(df[column_name].dropna())
            if pandas_dtype == 'mixed':
                raise TypeError(
                    'Unable to infer type of column {0!r}. Try instantiating '
                    'your table from the client with client.table('
                    "'my_table', schema={{{0!r}: <explicit type>}})".format(
                        column_name
                    )
                )

        ibis_dtype = dt.dtype(pandas_dtype)
        pairs.append((column_name, ibis_dtype))

    return Schema.from_tuples(pairs)
