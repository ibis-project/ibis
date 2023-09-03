from __future__ import annotations

import quartodoc.ast as qast
import toolz
from plum import dispatch
from quartodoc import MdRenderer


class Renderer(MdRenderer):
    style = "ibis"

    @dispatch
    def render(self, el: qast.ExampleCode):
        lines = el.value.splitlines()

        result = []

        prompt = ">>> "
        continuation = "... "
        skip_doctest = "doctest: +SKIP"
        expect_failure = "quartodoc: +EXPECTED_FAILURE"
        quartodoc_skip_doctest = "quartodoc: +SKIP"

        chunker = lambda line: line.startswith((prompt, continuation))

        for first, *rest in toolz.partitionby(chunker, lines):
            if first.startswith(prompt):
                if not (
                    quartodoc_skip_doctest in first
                    or skip_doctest in first
                    or any(
                        quartodoc_skip_doctest in line or skip_doctest in line
                        for line in rest
                    )
                ):
                    start, end = "{}"
                else:
                    start = end = ""

                result.append(f"```{start}python{end}")

                if expect_failure in first:
                    assert (
                        start and end
                    ), "expected failure should never occur alongside a skipped doctest example"
                    result.append("#| error: true")
                result.append(
                    first.replace(f"# {quartodoc_skip_doctest}", "")
                    .replace(quartodoc_skip_doctest, "")
                    .replace(f"# {expect_failure}", "")
                    .replace(expect_failure, "")
                )
                result.extend(rest)
                result.append("```\n")

        return "\n".join(result)
