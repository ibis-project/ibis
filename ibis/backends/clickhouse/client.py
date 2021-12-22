import re

import numpy as np
import pandas as pd

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")
base_typename_re = re.compile(r"(\w+)")


_clickhouse_dtypes = {
    'Null': dt.Null,
    'Nothing': dt.Null,
    'UInt8': dt.UInt8,
    'UInt16': dt.UInt16,
    'UInt32': dt.UInt32,
    'UInt64': dt.UInt64,
    'Int8': dt.Int8,
    'Int16': dt.Int16,
    'Int32': dt.Int32,
    'Int64': dt.Int64,
    'Float32': dt.Float32,
    'Float64': dt.Float64,
    'String': dt.String,
    'FixedString': dt.String,
    'Date': dt.Date,
    'DateTime': dt.Timestamp,
    'DateTime64': dt.Timestamp,
    'Array': dt.Array,
}
_ibis_dtypes = {v: k for k, v in _clickhouse_dtypes.items()}
_ibis_dtypes[dt.String] = 'String'
_ibis_dtypes[dt.Timestamp] = 'DateTime'


class ClickhouseDataType:

    __slots__ = 'typename', 'base_typename', 'nullable'

    def __init__(self, typename, nullable=False):
        m = base_typename_re.match(typename)
        self.base_typename = m.groups()[0]
        if self.base_typename not in _clickhouse_dtypes:
            raise com.UnsupportedBackendType(typename)
        self.typename = self.base_typename
        self.nullable = nullable

        if self.base_typename == 'Array':
            self.typename = typename

    def __str__(self):
        if self.nullable:
            return f'Nullable({self.typename})'
        else:
            return self.typename

    def __repr__(self):
        return f'<Clickhouse {str(self)}>'

    @classmethod
    def parse(cls, spec):
        # TODO(kszucs): spare parsing, depends on clickhouse-driver#22
        if spec.startswith('Nullable'):
            return cls(spec[9:-1], nullable=True)
        else:
            return cls(spec)

    def to_ibis(self):
        if self.base_typename != 'Array':
            return _clickhouse_dtypes[self.typename](nullable=self.nullable)

        sub_type = ClickhouseDataType(
            self.get_subname(self.typename)
        ).to_ibis()
        return dt.Array(value_type=sub_type)

    @staticmethod
    def get_subname(name: str) -> str:
        lbracket_pos = name.find('(')
        rbracket_pos = name.rfind(')')

        if lbracket_pos == -1 or rbracket_pos == -1:
            return ''

        subname = name[lbracket_pos + 1 : rbracket_pos]
        return subname

    @staticmethod
    def get_typename_from_ibis_dtype(dtype):
        if not isinstance(dtype, dt.Array):
            return _ibis_dtypes[type(dtype)]

        return 'Array({})'.format(
            ClickhouseDataType.get_typename_from_ibis_dtype(dtype.value_type)
        )

    @classmethod
    def from_ibis(cls, dtype, nullable=None):
        typename = ClickhouseDataType.get_typename_from_ibis_dtype(dtype)
        if nullable is None:
            nullable = dtype.nullable
        return cls(typename, nullable=nullable)


@dt.dtype.register(ClickhouseDataType)
def clickhouse_to_ibis_dtype(clickhouse_dtype):
    return clickhouse_dtype.to_ibis()


class ClickhouseTable(ir.TableExpr):
    """References a physical table in Clickhouse"""

    @property
    def _qualified_name(self):
        return self.op().args[0]

    @property
    def _unqualified_name(self):
        return self._match_name()[1]

    @property
    def _client(self):
        return self.op().args[2]

    def _match_name(self):
        m = fully_qualified_re.match(self._qualified_name)
        if not m:
            raise com.IbisError(
                'Cannot determine database name from {}'.format(
                    self._qualified_name
                )
            )
        db, quoted, unquoted = m.groups()
        return db, quoted or unquoted

    @property
    def _database(self):
        return self._match_name()[0]

    def invalidate_metadata(self):
        self._client.invalidate_metadata(self._qualified_name)

    def metadata(self):
        """
        Return parsed results of DESCRIBE FORMATTED statement

        Returns
        -------
        meta : TableMetadata
        """
        return self._client.describe_formatted(self._qualified_name)

    describe_formatted = metadata

    @property
    def name(self):
        return self.op().name

    def insert(self, obj, **kwargs):
        from .identifiers import quote_identifier

        schema = self.schema()

        assert isinstance(obj, pd.DataFrame)
        assert set(schema.names) >= set(obj.columns)

        columns = ', '.join(map(quote_identifier, obj.columns))
        query = 'INSERT INTO {table} ({columns}) VALUES'.format(
            table=self._qualified_name, columns=columns
        )

        # convert data columns with datetime64 pandas dtype to native date
        # because clickhouse-driver 0.0.10 does arithmetic operations on it
        obj = obj.copy()
        for col in obj.select_dtypes(include=[np.datetime64]):
            if isinstance(schema[col], dt.Date):
                obj[col] = obj[col].dt.date

        data = obj.to_dict('records')
        return self._client.con.execute(query, data, **kwargs)
