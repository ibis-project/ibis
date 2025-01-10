import pytest
from typing import Any

from ibis.common.typing import get_type_hints

class TestClass:
    a: int
    @property
    def b(self) -> str:
        return "test"
    @property
    def c(self) -> Any:
        raise AttributeError("Property 'c' raises AttributeError")

def test_get_type_hints_with_attribute_error():
    hints = get_type_hints(TestClass, include_properties=True)
    assert hints == {"a": int, "b": str}
