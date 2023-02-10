from __future__ import annotations

import importlib.resources
import json
import os
from typing import TYPE_CHECKING

import pooch

import ibis

if TYPE_CHECKING:
    import ibis.expr.types as ir


EXAMPLES = pooch.create(
    path=pooch.os_cache("ibis-framework"),
    # the trailing slash matters here
    base_url="https://storage.googleapis.com/ibis-examples/data/",
    version=ibis.__version__,
)
with importlib.resources.open_text(__name__, "registry.txt") as f:
    EXAMPLES.load_registry(f)

DESCRIPTIONS = json.loads(importlib.resources.read_text(__name__, "descriptions.json"))


class Example:
    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        self.key = key

    def __repr__(self) -> str:
        key = self.key
        description = self.__class__.__doc__
        return f"{self.__class__.__name__}({description!r}, key={key!r})"

    def fetch(self, **kwargs) -> ir.Table:
        return ibis.read_csv(EXAMPLES.fetch(self.key), **kwargs)


def __dir__() -> list[str]:
    return sorted(key.split(os.extsep, 1)[0] for key in EXAMPLES.registry.keys())


def __getattr__(name: str) -> Example:
    key = f"{name}.csv.gz"

    if key not in EXAMPLES.registry:
        raise AttributeError(name)

    description = DESCRIPTIONS.get(name, "No description available")

    example_class = type(name, (Example,), {"__slots__": (), "__doc__": description})
    example = example_class(key)
    setattr(ibis.examples, name, example)
    return example


__all__ = sorted(__dir__())  # noqa: PLE0605
