"""JSON value operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis.expr.operations as ops
from ibis.expr.types import Column, Scalar, Value

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class JSONValue(Value):
    def __getitem__(
        self, key: str | int | ir.StringValue | ir.IntegerValue
    ) -> JSONValue:
        """Access an JSON object's value or JSON array's element at `key`.

        Parameters
        ----------
        key
            Object field name or integer array index

        Returns
        -------
        JSONValue
            Element located at `key`

        Examples
        --------
        Construct a table with a JSON column

        >>> import json, ibis
        >>> ibis.options.interactive = True
        >>> rows = [{"js": json.dumps({"a": [i, 1]})} for i in range(2)]
        >>> t = ibis.memtable(rows, schema=ibis.schema(dict(js="json")))
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ js                   ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ json                 │
        ├──────────────────────┤
        │ {'a': [...]}         │
        │ {'a': [...]}         │
        └──────────────────────┘

        Extract the `"a"` field

        >>> t.js["a"]
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ JSONGetItem(js, 'a') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ json                 │
        ├──────────────────────┤
        │ [0, 1]               │
        │ [1, 1]               │
        └──────────────────────┘

        Extract the first element of the JSON array at `"a"`

        >>> t.js["a"][0]
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ JSONGetItem(JSONGetItem(js, 'a'), 0) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ json                                 │
        ├──────────────────────────────────────┤
        │ 0                                    │
        │ 1                                    │
        └──────────────────────────────────────┘

        Extract a non-existent field

        >>> t.js["a"]["foo"]
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ JSONGetItem(JSONGetItem(js, 'a'), 'foo') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ json                                     │
        ├──────────────────────────────────────────┤
        │ NULL                                     │
        │ NULL                                     │
        └──────────────────────────────────────────┘

        Try to extract an array element, returns `NULL`

        >>> t.js[20]
        ┏━━━━━━━━━━━━━━━━━━━━━┓
        ┃ JSONGetItem(js, 20) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━┩
        │ json                │
        ├─────────────────────┤
        │ NULL                │
        │ NULL                │
        └─────────────────────┘
        """
        return ops.JSONGetItem(self, key).to_expr()

    @property
    def map(self) -> ir.MapValue:
        """Cast JSON to a map of string to JSON.

        Use this property to unlock map functionality on JSON objects.

        Returns
        -------
        MapValue
            Map of string to JSON
        """
        return ops.ToJSONMap(self).to_expr()

    @property
    def array(self) -> ir.ArrayValue:
        """Cast JSON to an array of JSON.

        Use this property to unlock array functionality on JSON objects.

        Returns
        -------
        ArrayValue
            Array of JSON objects
        """
        return ops.ToJSONArray(self).to_expr()


@public
class JSONScalar(Scalar, JSONValue):
    pass


@public
class JSONColumn(Column, JSONValue):
    def __getitem__(
        self, key: str | int | ir.StringValue | ir.IntegerValue
    ) -> JSONColumn:
        return JSONValue.__getitem__(self, key)
