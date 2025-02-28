from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Optional

import ibis
from ibis.common.grounds import Concrete

try:
    import importlib_resources as resources
except ImportError:
    from importlib import resources

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from ibis.backends import BaseBackend


class Example(Concrete):
    name: str
    help: Optional[str]

    def fetch(
        self,
        *,
        table_name: str | None = None,
        backend: BaseBackend | None = None,
    ) -> ir.Table:
        if backend is None:
            backend = ibis.get_backend()

        name = self.name

        if table_name is None:
            table_name = name

        board = _get_board()
        (path,) = board.pin_download(name)
        return backend._load_example(path=path, table_name=table_name)


_FETCH_DOCSTRING_TEMPLATE = """\
Fetch the {name} example.

Parameters
----------
table_name
    The table name to use, defaults to a generated table name.
backend
    The backend to load the example into. Defaults to the default backend.

Returns
-------
ir.Table
    Table expression

Examples
--------
>>> import ibis
>>> t = ibis.examples.{name}.fetch()
"""

_BUCKET = "ibis-pins"


@functools.cache
def _get_metadata():
    return json.loads(resources.files(__name__).joinpath("metadata.json").read_text())


@functools.cache
def _get_board():
    import pins

    return pins.board(
        "gcs", _BUCKET, storage_options={"cache_timeout": 0, "token": "anon"}
    )


@functools.cache
def __dir__() -> list[str]:
    return sorted(_get_metadata().keys())


class Zones(Concrete):
    name: str
    help: Optional[str]

    def fetch(
        self,
        *,
        table_name: str | None = None,
        backend: BaseBackend | None = None,
    ) -> ir.Table:
        if backend is None:
            backend = ibis.get_backend()

        name = self.name

        if table_name is None:
            table_name = name

        board = _get_board()

        (path,) = board.pin_download(name)
        return backend.read_geo(path)


zones = Zones("zones", help="Taxi zones in New York City (EPSG:2263)")


def __getattr__(name: str) -> Example:
    try:
        meta = _get_metadata()

        description = meta[name].get("description")

        fields = {"__doc__": description} if description is not None else {}

        example_class = type(name, (Example,), fields)
        example_class.fetch.__doc__ = _FETCH_DOCSTRING_TEMPLATE.format(name=name)

        example = example_class(name=name, help=description)
        setattr(ibis.examples, name, example)
    except Exception as e:
        raise AttributeError(name) from e
    else:
        return example
