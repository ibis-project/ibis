from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pooch

import ibis
from ibis.common.grounds import Concrete

try:
    import importlib_resources as resources
except ImportError:
    from importlib import resources

if TYPE_CHECKING:
    import ibis.expr.types as ir


EXAMPLES = pooch.create(
    path=pooch.os_cache("ibis-framework"),
    # the trailing slash matters here
    base_url="https://storage.googleapis.com/ibis-examples/data/",
    version=ibis.__version__,
)
with resources.files(__name__).joinpath("registry.txt").open(mode="r") as _f:
    EXAMPLES.load_registry(_f)

DESCRIPTIONS = json.loads(
    resources.files(__name__).joinpath("descriptions.json").read_text()
)


class Example(Concrete):
    descr: str
    key: str

    def fetch(self, **kwargs) -> ir.Table:
        return ibis.read_csv(EXAMPLES.fetch(self.key), **kwargs)


def __dir__() -> list[str]:
    return sorted(key.split(os.extsep, 1)[0] for key in EXAMPLES.registry.keys())


def __getattr__(name: str) -> Example:
    key = f"{name}.csv.gz"

    if key not in EXAMPLES.registry:
        raise AttributeError(name)

    description = DESCRIPTIONS.get(name, "No description available")

    example_class = type(name, (Example,), {"__doc__": description})
    example = example_class(descr=description, key=key)
    setattr(ibis.examples, name, example)
    return example


__all__ = sorted(__dir__())  # noqa: PLE0605
