from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import pooch

import ibis
from ibis.common.grounds import Concrete

try:
    import importlib_resources as resources
except ImportError:
    from importlib import resources

if TYPE_CHECKING:
    import ibis.expr.types as ir


_EXAMPLES = pooch.create(
    path=pooch.os_cache("ibis-framework"),
    # the trailing slash matters here
    base_url="https://storage.googleapis.com/ibis-examples/data/",
    version=ibis.__version__,
)
with resources.files(__name__).joinpath("registry.txt").open(mode="r") as _f:
    _EXAMPLES.load_registry(_f)

_METADATA = json.loads(resources.files(__name__).joinpath("metadata.json").read_text())

_READER_FUNCS = {"csv": "read_csv", "csv.gz": "read_csv", "parquet": "read_parquet"}


class Example(Concrete):
    descr: Optional[str]  # noqa: UP007
    key: str
    reader: str

    def fetch(self, **kwargs: Any) -> ir.Table:
        reader = getattr(ibis, self.reader)
        return reader(_EXAMPLES.fetch(self.key, progressbar=True), **kwargs)


def __dir__() -> list[str]:
    return sorted(_METADATA.keys())


def _make_fetch_docstring(*, name: str, reader: str):
    return f"""Fetch the {name} example.
Parameters
----------
kwargs
    Same as the arguments for [`ibis.{reader}`][ibis.{reader}]

Returns
-------
ir.Table
    Table expression

Examples
--------
>>> import ibis
>>> t = ibis.examples.{name}.fetch()
"""


def __getattr__(name: str) -> Example:
    spec = _METADATA.get(name, {})

    if (key := spec.get("key")) is None:
        raise AttributeError(name)

    description = spec.get("description")

    _, ext = key.split(os.extsep, maxsplit=1)
    reader = _READER_FUNCS[ext]

    fields = {"__doc__": description} if description is not None else {}

    example_class = type(name, (Example,), fields)
    example_class.fetch.__doc__ = _make_fetch_docstring(name=name, reader=reader)
    example = example_class(descr=description, key=key, reader=reader)
    setattr(ibis.examples, name, example)
    return example
