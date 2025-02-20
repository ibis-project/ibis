"""JSON value operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.expr.types import Column, Scalar, Value

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class JSONValue(Value):
    """A json-like collection with dynamic keys and values.

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
        """
        return ops.JSONGetItem(self, key).to_expr()

    def unwrap_as(self, dtype: dt.DataType | str) -> ir.Value:
        """Unwrap JSON into a specific data type.

        Returns
        -------
        Value
            An Ibis expression of a more specific type than JSON

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> data = {
        ...     "jstring": ['"a"', '""', None, "null"],
        ...     "jbool": ["true", "false", "null", None],
        ...     "jint": ["1", "null", None, "2"],
        ...     "jfloat": ["42.42", None, "null", "37.37"],
        ...     "jmap": ['{"a": 1}', "null", None, "{}"],
        ...     "jarray": ["[]", "null", None, '[{},"1",2]'],
        ... }
        >>> t = ibis.memtable(data, schema=dict.fromkeys(data.keys(), "json"))
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ jstring              ┃ jbool                ┃ jint                 ┃ … ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ json                 │ json                 │ json                 │ … │
        ├──────────────────────┼──────────────────────┼──────────────────────┼───┤
        │ 'a'                  │ True                 │ 1                    │ … │
        │ ''                   │ False                │ None                 │ … │
        │ NULL                 │ None                 │ NULL                 │ … │
        │ None                 │ NULL                 │ 2                    │ … │
        └──────────────────────┴──────────────────────┴──────────────────────┴───┘
        >>> t.select(unwrapped=t.jstring.unwrap_as(str), original=t.jstring)
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ unwrapped ┃ original             ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ string    │ json                 │
        ├───────────┼──────────────────────┤
        │ a         │ 'a'                  │
        │ ~         │ ''                   │
        │ NULL      │ NULL                 │
        │ NULL      │ None                 │
        └───────────┴──────────────────────┘
        >>> t.select(unwrapped=t.jbool.unwrap_as("bool"), original=t.jbool)
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ unwrapped ┃ original             ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean   │ json                 │
        ├───────────┼──────────────────────┤
        │ True      │ True                 │
        │ False     │ False                │
        │ NULL      │ None                 │
        │ NULL      │ NULL                 │
        └───────────┴──────────────────────┘
        >>> t.select(
        ...     unwrapped_int64=t.jint.unwrap_as("int64"),
        ...     unwrapped_int32=t.jint.unwrap_as("int32"),
        ...     original=t.jint,
        ... )
        ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ unwrapped_int64 ┃ unwrapped_int32 ┃ original             ┃
        ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64           │ int32           │ json                 │
        ├─────────────────┼─────────────────┼──────────────────────┤
        │               1 │               1 │ 1                    │
        │            NULL │            NULL │ None                 │
        │            NULL │            NULL │ NULL                 │
        │               2 │               2 │ 2                    │
        └─────────────────┴─────────────────┴──────────────────────┘

        You can cast to a more specific type than the types available in standards-compliant JSON.

        Here's an example of casting JSON numbers to `float32`:

        >>> t.select(unwrapped=t.jfloat.unwrap_as("float32"), original=t.jfloat)
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ unwrapped ┃ original             ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ float32   │ json                 │
        ├───────────┼──────────────────────┤
        │ 42.419998 │ 42.42                │
        │      NULL │ NULL                 │
        │      NULL │ None                 │
        │ 37.369999 │ 37.37                │
        └───────────┴──────────────────────┘

        You can cast JSON objects to a more specific `map` type:

        >>> t.select(unwrapped=t.jmap.unwrap_as("map<string, int>"), original=t.jmap)
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ unwrapped            ┃ original             ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ map<string, int64>   │ json                 │
        ├──────────────────────┼──────────────────────┤
        │ {'a': 1}             │ {'a': 1}             │
        │ NULL                 │ None                 │
        │ NULL                 │ NULL                 │
        │ {}                   │ {}                   │
        └──────────────────────┴──────────────────────┘

        You can cast JSON arrays to an array type as well. In this case the
        array values don't have a single element type so we cast to
        `array<json>`.

        >>> t.select(unwrapped=t.jarray.unwrap_as("array<json>"), original=t.jarray)
        ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ unwrapped             ┃ original             ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<json>           │ json                 │
        ├───────────────────────┼──────────────────────┤
        │ []                    │ []                   │
        │ NULL                  │ None                 │
        │ NULL                  │ NULL                 │
        │ ['{}', '"1"', ... +1] │ [{...}, '1', ... +1] │
        └───────────────────────┴──────────────────────┘

        See Also
        --------
        [`JSONValue.str`](#ibis.expr.types.json.JSONValue.str)
        [`JSONValue.int`](#ibis.expr.types.json.JSONValue.int)
        [`JSONValue.float`](#ibis.expr.types.json.JSONValue.float)
        [`JSONValue.bool`](#ibis.expr.types.json.JSONValue.bool)
        [`JSONValue.map`](#ibis.expr.types.json.JSONValue.map)
        [`JSONValue.array`](#ibis.expr.types.json.JSONValue.array)
        [`Value.cast`](#ibis.expr.types.generic.Value.cast)
        """
        dtype = dt.dtype(dtype)
        if dtype.is_string():
            return self.str
        elif dtype.is_boolean():
            return self.bool
        elif dtype.is_integer():
            i = self.int
            return i.cast(dtype) if i.type() != dtype else i
        elif dtype.is_floating():
            f = self.float
            return f.cast(dtype) if f.type() != dtype else f
        elif dtype.is_map():
            m = self.map
            return m.cast(dtype) if m.type() != dtype else m
        elif dtype.is_array():
            a = self.array
            return a.cast(dtype) if a.type() != dtype else a
        else:
            raise exc.IbisTypeError(
                f"Data type {dtype} is unsupported for unwrapping JSON values. Supported "
                "data types are strings, integers, floats, booleans, maps, and arrays."
            )

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

    @property
    def int(self) -> ir.IntegerValue:
        """Unwrap a JSON value into a backend-native int.

        Any non-float JSON values are returned as `NULL`.

        Examples
        --------
        >>> import json, ibis
        >>> ibis.options.interactive = True
        >>> data = [
        ...     {"name": "Alice", "json_data": '{"last_name":"Smith","age":40}'},
        ...     {"name": "Bob", "json_data": '{"last_name":"Jones", "age":39}'},
        ...     {"name": "Charlie", "json_data": '{"last_name":"Davies","age":54}'},
        ... ]
        >>> t = ibis.memtable(data, schema={"name": "string", "json_data": "json"})
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ name    ┃ json_data                          ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string  │ json                               │
        ├─────────┼────────────────────────────────────┤
        │ Alice   │ {'last_name': 'Smith', 'age': 40}  │
        │ Bob     │ {'last_name': 'Jones', 'age': 39}  │
        │ Charlie │ {'last_name': 'Davies', 'age': 54} │
        └─────────┴────────────────────────────────────┘
        >>> t.mutate(age=t.json_data["age"].int)
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
        ┃ name    ┃ json_data                          ┃ age   ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
        │ string  │ json                               │ int64 │
        ├─────────┼────────────────────────────────────┼───────┤
        │ Alice   │ {'last_name': 'Smith', 'age': 40}  │    40 │
        │ Bob     │ {'last_name': 'Jones', 'age': 39}  │    39 │
        │ Charlie │ {'last_name': 'Davies', 'age': 54} │    54 │
        └─────────┴────────────────────────────────────┴───────┘
        """
        return ops.UnwrapJSONInt64(self).to_expr()

    @property
    def float(self) -> ir.FloatingValue:
        """Unwrap a JSON value into a backend-native float.

        Any non-float JSON values are returned as `NULL`.

        ::: {.callout-warning}
        ## The `float` property is lax with respect to integers

        The `float` property will attempt to coerce integers to floating point numbers.
        :::

        Examples
        --------
        >>> import json, ibis
        >>> ibis.options.interactive = True
        >>> data = [
        ...     {"name": "Alice", "json_data": '{"last_name":"Smith","salary":42.42}'},
        ...     {"name": "Bob", "json_data": '{"last_name":"Jones", "salary":37.37}'},
        ...     {"name": "Charlie", "json_data": '{"last_name":"Davies","salary":"NA"}'},
        ...     {"name": "Joan", "json_data": '{"last_name":"Davies","salary":78}'},
        ... ]
        >>> t = ibis.memtable(data, schema={"name": "string", "json_data": "json"})
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ name    ┃ json_data                               ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string  │ json                                    │
        ├─────────┼─────────────────────────────────────────┤
        │ Alice   │ {'last_name': 'Smith', 'salary': 42.42} │
        │ Bob     │ {'last_name': 'Jones', 'salary': 37.37} │
        │ Charlie │ {'last_name': 'Davies', 'salary': 'NA'} │
        │ Joan    │ {'last_name': 'Davies', 'salary': 78}   │
        └─────────┴─────────────────────────────────────────┘
        >>> t.mutate(salary=t.json_data["salary"].float)
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ name    ┃ json_data                               ┃ salary  ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
        │ string  │ json                                    │ float64 │
        ├─────────┼─────────────────────────────────────────┼─────────┤
        │ Alice   │ {'last_name': 'Smith', 'salary': 42.42} │   42.42 │
        │ Bob     │ {'last_name': 'Jones', 'salary': 37.37} │   37.37 │
        │ Charlie │ {'last_name': 'Davies', 'salary': 'NA'} │    NULL │
        │ Joan    │ {'last_name': 'Davies', 'salary': 78}   │   78.00 │
        └─────────┴─────────────────────────────────────────┴─────────┘
        """
        return ops.UnwrapJSONFloat64(self).to_expr()

    @property
    def bool(self) -> ir.BooleanValue:
        """Unwrap a JSON value into a backend-native boolean.

        Any non-boolean JSON values are returned as `NULL`.

        Examples
        --------
        >>> import json, ibis
        >>> ibis.options.interactive = True
        >>> data = [
        ...     {"name": "Alice", "json_data": '{"last_name":"Smith","is_bot":false}'},
        ...     {"name": "Bob", "json_data": '{"last_name":"Jones","is_bot":true}'},
        ...     {"name": "Charlie", "json_data": '{"last_name":"Davies","is_bot":false}'},
        ... ]
        >>> t = ibis.memtable(data, schema={"name": "string", "json_data": "json"})
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ name    ┃ json_data                                ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string  │ json                                     │
        ├─────────┼──────────────────────────────────────────┤
        │ Alice   │ {'last_name': 'Smith', 'is_bot': False}  │
        │ Bob     │ {'last_name': 'Jones', 'is_bot': True}   │
        │ Charlie │ {'last_name': 'Davies', 'is_bot': False} │
        └─────────┴──────────────────────────────────────────┘
        >>> t.mutate(is_bot=t.json_data["is_bot"].bool)
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ name    ┃ json_data                                ┃ is_bot  ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
        │ string  │ json                                     │ boolean │
        ├─────────┼──────────────────────────────────────────┼─────────┤
        │ Alice   │ {'last_name': 'Smith', 'is_bot': False}  │ False   │
        │ Bob     │ {'last_name': 'Jones', 'is_bot': True}   │ True    │
        │ Charlie │ {'last_name': 'Davies', 'is_bot': False} │ False   │
        └─────────┴──────────────────────────────────────────┴─────────┘
        """
        return ops.UnwrapJSONBoolean(self).to_expr()

    @property
    def str(self) -> ir.StringValue:
        """Unwrap a JSON string into a backend-native string.

        Any non-string JSON values are returned as `NULL`.

        Returns
        -------
        StringValue
            A string expression

        Examples
        --------
        >>> import json, ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {"js": ['"a"', '"b"', "1", "{}", '[{"a": 1}]']},
        ...     schema=ibis.schema(dict(js="json")),
        ... )
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ js                   ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ json                 │
        ├──────────────────────┤
        │ 'a'                  │
        │ 'b'                  │
        │ 1                    │
        │ {}                   │
        │ [{...}]              │
        └──────────────────────┘
        >>> t.js.str
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ UnwrapJSONString(js) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ string               │
        ├──────────────────────┤
        │ a                    │
        │ b                    │
        │ NULL                 │
        │ NULL                 │
        │ NULL                 │
        └──────────────────────┘

        Note the difference between `.string` and `.cast("string")`.

        The latter preserves quotes for JSON string values and returns a valid
        JSON string.

        >>> t.js.cast("string")
        ┏━━━━━━━━━━━━━━━━━━┓
        ┃ Cast(js, string) ┃
        ┡━━━━━━━━━━━━━━━━━━┩
        │ string           │
        ├──────────────────┤
        │ "a"              │
        │ "b"              │
        │ 1                │
        │ {}               │
        │ [{"a": 1}]       │
        └──────────────────┘

        Here's a more complex example with a table containing a JSON column
        with nested fields.

        >>> data = [
        ...     {"name": "Alice", "json_data": '{"last_name":"Smith"}'},
        ...     {"name": "Bob", "json_data": '{"last_name":"Jones"}'},
        ...     {"name": "Charlie", "json_data": '{"last_name":"Davies"}'},
        ... ]
        >>> t = ibis.memtable(data, schema={"name": "string", "json_data": "json"})
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ name    ┃ json_data               ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string  │ json                    │
        ├─────────┼─────────────────────────┤
        │ Alice   │ {'last_name': 'Smith'}  │
        │ Bob     │ {'last_name': 'Jones'}  │
        │ Charlie │ {'last_name': 'Davies'} │
        └─────────┴─────────────────────────┘
        >>> t.mutate(last_name=t.json_data["last_name"].str)
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
        ┃ name    ┃ json_data               ┃ last_name ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
        │ string  │ json                    │ string    │
        ├─────────┼─────────────────────────┼───────────┤
        │ Alice   │ {'last_name': 'Smith'}  │ Smith     │
        │ Bob     │ {'last_name': 'Jones'}  │ Jones     │
        │ Charlie │ {'last_name': 'Davies'} │ Davies    │
        └─────────┴─────────────────────────┴───────────┘
        """
        return ops.UnwrapJSONString(self).to_expr()


@public
class JSONScalar(Scalar, JSONValue):
    pass


@public
class JSONColumn(Column, JSONValue):
    def __getitem__(
        self, key: str | int | ir.StringValue | ir.IntegerValue
    ) -> JSONColumn:
        return JSONValue.__getitem__(self, key)
