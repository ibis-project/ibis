import collections

from multipledispatch import Dispatcher

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.util as util


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

        self._name_locs = dict((v, i) for i, v in enumerate(self.names))

        if len(self._name_locs) < len(self.names):
            duplicate_names = list(self.names)
            for v in self._name_locs.keys():
                duplicate_names.remove(v)
            raise com.IntegrityError(
                'Duplicate column name(s): {}'.format(duplicate_names)
            )

    def __repr__(self):
        space = 2 + max(map(len, self.names), default=0)
        return "ibis.Schema {{{}\n}}".format(
            util.indent(
                ''.join(
                    '\n{}{}'.format(name.ljust(space), str(type))
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


class HasSchema:

    """
    Base class representing a structured dataset with a well-defined
    schema.

    Base implementation is for tables that do not reference a particular
    concrete dataset or database table.
    """

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
