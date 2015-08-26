# Copyright 2015 Cloudera Inc.
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

import sqlalchemy as sa
import ibis.expr.datatypes as dt


_ibis_type_to_sqla = {
    dt.Int8: sa.types.SmallInteger,
    dt.Int16: sa.types.SmallInteger,
    dt.Int32: sa.types.Integer,
    dt.Int64: sa.types.BigInteger,

    # Mantissa-based
    dt.Float: sa.types.Float(precision=24),
    dt.Double: sa.types.Float(precision=53),

    dt.Boolean: sa.types.Boolean,

    dt.String: sa.types.String,

    dt.Timestamp: sa.types.DateTime,

    dt.Decimal: sa.types.NUMERIC,
}

_sqla_overrides = {
    sa.types.SmallInteger: dt.Int16,
    sa.types.FLOAT: dt.Double,
    sa.types.REAL: dt.Double,
    sa.types.NullType: dt.String,
}

_sqla_type_to_ibis = dict((v, k) for k, v in
                          _ibis_type_to_sqla.items())
_sqla_type_to_ibis.update(_sqla_overrides)


def schema_from_table(table):
    names = table.columns.keys()

    types = []
    for c in table.columns.values():
        type_class = type(c.type)

        if c.type in _sqla_type_to_ibis:
            ibis_class = _sqla_type_to_ibis[c.type]
        elif type_class in _sqla_type_to_ibis:
            ibis_class = _sqla_type_to_ibis[type_class]
        else:
            raise NotImplementedError(c.type)

        t = ibis_class(c.nullable)
        types.append(t)

    return dt.Schema(names, types)
