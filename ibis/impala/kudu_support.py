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

from ibis.expr.api import schema
import ibis.expr.datatypes as dt
import kudu


_kudu_type_to_ibis_typeclass = {
    'int8': dt.Int8,
    'int16': dt.Int16,
    'int32': dt.Int32,
    'int64': dt.Int64,
    'float': dt.Float,
    'double': dt.Double,
    'bool': dt.Boolean,
    'string': dt.String,
    'timestamp': dt.Timestamp
}


def schema_kudu_to_ibis(kschema):
    ibis_types = []
    for i in range(len(kschema)):
        col = kschema[i]

        typeclass = _kudu_type_to_ibis_typeclass[col.type.name]
        itype = typeclass(col.nullable)

        ibis_types.append((col.name, itype))

    return schema(ibis_types)
