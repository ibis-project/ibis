from __future__ import annotations

import ast
from textwrap import dedent

import griffe
import quartodoc as qd
import toolz
from plum import dispatch

from ibis.util import (
    BACKEND_SENSITIVE_MSG,
    EXPERIMENTAL_MSG,
    append_admonition,
    deprecated_msg,
)


def _keyword_arguments(value: griffe.Expr) -> dict[str, str]:
    """Statically evaluate the keyword arguments of a decorator call."""
    if not isinstance(value, griffe.ExprCall):
        return {}
    return {
        argument.name: ast.literal_eval(str(argument.value))
        for argument in value.arguments
        if isinstance(argument, griffe.ExprKeyword)
    }


def _admonition_arguments(decorator: griffe.Decorator, el) -> dict[str, str] | None:
    """Return `append_admonition` arguments for `decorator`, if it adds one."""
    kwargs = _keyword_arguments(decorator.value)
    path = decorator.value.canonical_path

    if path == "ibis.util.experimental":
        return {"msg": EXPERIMENTAL_MSG}
    elif path == "ibis.util.deprecated":
        try:
            qualname = el.path.removeprefix(f"{el.module.path}.")
        except (AttributeError, ValueError):
            qualname = el.name
        return {"msg": f"DEPRECATED: {deprecated_msg(qualname, **kwargs)}"}
    elif path == "ibis.util.backend_sensitive":
        return {
            "msg": kwargs.get("msg", BACKEND_SENSITIVE_MSG),
            "body": kwargs.get("why", ""),
            "kind": "note",
        }
    else:
        return None


def apply_admonitions(el) -> None:
    """Rebuild the admonitions that ibis's decorators add to `__doc__`.

    The `experimental`, `deprecated` and `backend_sensitive` decorators inject
    their admonitions when the module is imported, but objects are collected by
    parsing source, so none of that is visible to the docs build. The
    decorators themselves *are* visible, so reproduce their output from them.
    """
    # decorators are applied bottom-up, and each admonition is inserted directly
    # after the summary line, so replay them in the same order to match `help()`
    for decorator in reversed(getattr(el, "decorators", ())):
        if (kwargs := _admonition_arguments(decorator, el)) is None:
            continue

        docstring = el.docstring

        # entries collected dynamically already have the decorator's admonition,
        # because they read the docstring the decorator built at import time
        if docstring is not None and f"## {kwargs['msg']}" in docstring.value:
            continue

        value = append_admonition(docstring.value if docstring else None, **kwargs)

        if docstring is None:
            el.docstring = griffe.Docstring(value, parent=el)
        else:
            docstring.value = value
            # `parsed` is a cached property, so drop any parse of the old value
            docstring.__dict__.pop("parsed", None)


class Renderer(qd.MdRenderer):
    style = "ibis"

    @dispatch
    def render(self, el: griffe.Object | griffe.Alias):
        apply_admonitions(el)
        return super().render(el)

    @dispatch
    def render(self, el: qd.ast.ExampleCode) -> str:  # noqa: F811
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
