from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING

from quartodoc import MdRenderer, get_object

if TYPE_CHECKING:
    from collections.abc import Iterator


@cache
def get_renderer(level: int) -> MdRenderer:
    return MdRenderer(header_level=level)


@cache
def get_backend(backend: str):
    return get_object(f"ibis.backends.{backend}", "Backend")


def get_callable(obj, name):
    try:
        return obj.get_member(name)
    except KeyError:
        return obj.functions[name]


def render_method(*, member, renderer: MdRenderer) -> Iterator[str]:
    yield f"{'#' * renderer.crnt_header_level} {member.name} {{ #{member.path} }}"
    yield renderer.render(member)


def render_methods(obj, *methods: str, level: int) -> None:
    renderer = get_renderer(level)
    get = partial(get_callable, obj)
    print(  # noqa: T201
        "\n".join(
            line
            for member in map(get, methods)
            for line in render_method(member=member, renderer=renderer)
        )
    )


def render_do_connect(backend, level: int = 4) -> None:
    render_methods(get_backend(backend), "do_connect", level=level)
