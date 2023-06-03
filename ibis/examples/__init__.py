from __future__ import annotations

from typing import TYPE_CHECKING

import ibis

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from ibis.backends.base import BaseBackend

# loaded lazily
_EXAMPLES = _METADATA = None

# These backends load the data directly using `read_csv`/`read_parquet`. All
# other backends load the data using pyarrow, then passing it off to
# `create_table`.
_DIRECT_BACKENDS = frozenset({"duckdb", "polars"})


class Example:
    __slots__ = "descr", "key"

    def __init__(self, descr: str | None, key: str) -> None:
        self.descr = descr
        self.key = key

    def __hash__(self) -> int:
        return hash((type(self), self.descr, self.key))

    def fetch(
        self,
        *,
        table_name: str | None = None,
        backend: BaseBackend | None = None,
    ) -> ir.Table:
        path = _EXAMPLES.fetch(self.key, progressbar=True)

        if backend is None:
            backend = ibis.get_backend()

        if table_name is None:
            table_name = ibis.util.gen_name(f"examples_{type(self).__name__}")

        if backend.name in _DIRECT_BACKENDS:
            # Read directly into these backends. This helps reduce memory
            # usage, making the larger example datasets easier to work with.
            if path.endswith(".parquet"):
                return backend.read_parquet(path, table_name=table_name)
            else:
                return backend.read_csv(path, table_name=table_name)
        else:
            if path.endswith(".parquet"):
                import pyarrow.parquet

                table = pyarrow.parquet.read_table(path)
            else:
                import pyarrow.csv

                # The convert options lets pyarrow treat empty strings as null for
                # string columns, but not quoted empty strings.
                table = pyarrow.csv.read_csv(
                    path,
                    convert_options=pyarrow.csv.ConvertOptions(
                        strings_can_be_null=True,
                        quoted_strings_can_be_null=False,
                    ),
                )

                # All null columns are inferred as null-type, but not all
                # backends support null-type columns. Cast to an all-null
                # string column instead.
                for i, field in enumerate(table.schema):
                    if pyarrow.types.is_null(field.type):
                        table = table.set_column(i, field.name, table[i].cast("string"))

            # TODO: It should be possible to avoid this memtable call, once all
            # backends support passing a `pyarrow.Table` to `create_table`
            # directly.
            obj = ibis.memtable(table)
            return backend.create_table(table_name, obj, temp=True, overwrite=True)


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


def _load_metadata():
    try:
        import importlib_resources as resources
    except ImportError:
        from importlib import resources

    global _EXAMPLES, _METADATA  # noqa: PLW0603

    if _EXAMPLES is None:
        import pooch

        _EXAMPLES = pooch.create(
            path=pooch.os_cache("ibis-framework"),
            # the trailing slash matters here
            base_url="https://storage.googleapis.com/ibis-examples/data/",
            version=ibis.__version__,
            env="IBIS_EXAMPLES_DATA",
        )
        with resources.files(__name__).joinpath("registry.txt").open(mode="r") as _f:
            _EXAMPLES.load_registry(_f)

    if _METADATA is None:
        import json

        _METADATA = json.loads(
            resources.files(__name__).joinpath("metadata.json").read_text()
        )


def __dir__() -> list[str]:
    _load_metadata()
    return sorted(_METADATA.keys())


def __getattr__(name: str) -> Example:
    _load_metadata()

    spec = _METADATA.get(name, {})

    if (key := spec.get("key")) is None:
        raise AttributeError(name)

    description = spec.get("description")

    fields = {"__doc__": description} if description is not None else {}

    example_class = type(name, (Example,), fields)
    example_class.fetch.__doc__ = _FETCH_DOCSTRING_TEMPLATE.format(name=name)
    example = example_class(descr=description, key=key)
    setattr(ibis.examples, name, example)
    return example
