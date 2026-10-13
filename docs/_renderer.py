from __future__ import annotations

import ast
from textwrap import dedent

import griffe
import quartodoc as qd
import toolz
from plum import dispatch

from ibis.util import import_object

# decorators that document a callable by adding an admonition to its `__doc__`
_ADMONITION_DECORATORS = frozenset(
    ("ibis.util.backend_sensitive", "ibis.util.deprecated", "ibis.util.experimental")
)


def _decorated_docstring(
    decorator: griffe.Decorator, obj: griffe.Object, docstring: str | None
) -> str:
    """Return `docstring` as `decorator` would leave it on `obj` at import time."""
    decorate = import_object(decorator.value.canonical_path)
    if isinstance(decorator.value, griffe.ExprCall):
        decorate = decorate(
            **{
                argument.name: ast.literal_eval(str(argument.value))
                for argument in decorator.value.arguments
                if isinstance(argument, griffe.ExprKeyword)
            }
        )

    def stand_in(): ...

    stand_in.__doc__ = docstring
    stand_in.__qualname__ = obj.path.removeprefix(f"{obj.module.path}.")
    return decorate(stand_in).__doc__


def apply_admonitions(obj: griffe.Object | griffe.Alias) -> None:
    """Add the admonitions that ibis's decorators add to `__doc__` at import time.

    Objects are collected by parsing source, so the docs build never sees what
    the decorators do to `__doc__`. It does see the decorators, so apply them to
    a stand-in carrying the parsed docstring.
    """
    if obj.is_alias:
        obj = obj.final_target

    old = obj.docstring
    original = value = old.value if old is not None else None

    # decorators apply bottom-up, so replay them in the same order
    for decorator in reversed(getattr(obj, "decorators", ())):
        if decorator.value.canonical_path not in _ADMONITION_DECORATORS:
            continue

        # skip docstrings that already have the admonition, either because they
        # were collected dynamically or because `obj` was already rendered
        if value and _decorated_docstring(decorator, obj, None) in value:
            continue

        value = _decorated_docstring(decorator, obj, value)

    if value != original:
        obj.docstring = griffe.Docstring(
            value,
            lineno=getattr(old, "lineno", None),
            endlineno=getattr(old, "endlineno", None),
            parent=obj,
            parser=getattr(old, "parser", None),
            parser_options=getattr(old, "parser_options", None),
        )


class Renderer(qd.MdRenderer):
    style = "ibis"

    @dispatch
    def render(self, el: qd.ast.ExampleCode) -> str:
        lines = el.value.splitlines()

        result = []

        prompt = ">>> "
        continuation = "... "

        skip_doctest = "doctest: +SKIP"
        expect_failure = "quartodoc: +EXPECTED_FAILURE"
        quartodoc_skip_doctest = "quartodoc: +SKIP"

        chunker = lambda line: line.startswith((prompt, continuation))
        should_skip = lambda line: (
            quartodoc_skip_doctest in line or skip_doctest in line
        )

        for first, *rest in toolz.partitionby(chunker, lines):
            # only attempt to execute or render code blocks that start with the
            # >>> prompt
            if first.startswith(prompt):
                # check whether to skip execution and if so, render the code
                # block as `python` (not `{python}`) if it's marked with
                # skip_doctest, expect_failure or quartodoc_skip_doctest
                if skipped := (should_skip(first) or any(map(should_skip, rest))):
                    start = end = ""
                else:
                    start, end = "{}"
                    result.append(
                        dedent(
                            """
                            ```{python}
                            #| echo: false

                            import ibis
                            ibis.options.interactive = True
                            ```
                            """
                        )
                    )

                result.append(f"```{start}python{end}")

                # if we expect failures, don't fail the notebook execution and
                # render the error message
                if expect_failure in first or any(
                    expect_failure in line for line in rest
                ):
                    assert start and end, (
                        "expected failure should never occur alongside a skipped doctest example"
                    )
                    result.append("#| error: true")

                # remove the quartodoc markers from the rendered code
                result.append(
                    first.removeprefix(prompt)
                    .replace(f"# {quartodoc_skip_doctest}", "")
                    .replace(quartodoc_skip_doctest, "")
                    .replace(f"# {expect_failure}", "")
                    .replace(expect_failure, "")
                )
                result.extend(
                    line.removeprefix(prompt).removeprefix(continuation)
                    for line in rest
                )
                result.append("```\n")

                if not skipped:
                    result.append(
                        dedent(
                            """
                            ```{python}
                            #| echo: false
                            ibis.options.interactive = False
                            ```
                            """
                        )
                    )

        return "\n".join(result)

    @dispatch
    def render(self, el: griffe.Object | griffe.Alias) -> str:  # noqa: F811
        apply_admonitions(el)
        return super().render(el)
