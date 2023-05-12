import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.common.exceptions import IntegrityError


def test_schema_from_names_types():
    s = ibis.schema(names=["a"], types=["array<float64>"])
    assert s == sch.Schema(dict(a="array<float64>"))


def test_schema_from_names_and_types_length_must_match():
    msg = "Schema names and types must have the same length"
    with pytest.raises(ValueError, match=msg):
        ibis.schema(names=["a", "b"], types=["int", "str", "float"])

    schema = ibis.schema(names=["a", "b"], types=["int", "str"])

    assert isinstance(schema, sch.Schema)
    assert schema.names == ("a", "b")
    assert schema.types == (dt.int64, dt.string)


def test_schema_from_names_and_typesield_names():
    msg = "Duplicate column name"
    with pytest.raises(IntegrityError, match=msg):
        ibis.schema(names=["a", "a"], types=["int", "str"])
