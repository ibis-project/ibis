from __future__ import annotations

import quartodoc as qd
import toolz
from plum import dispatch


class Renderer(qd.MdRenderer):
    style = "ibis"

    @dispatch
    def render(self, el: qd.ast.ExampleCode) -> str:
        lines = el.value.splitlines()

        result = []

        prompt = ">>> "
        continuation = "..."

        skip_doctest = "doctest: +SKIP"
        expect_failure = "quartodoc: +EXPECTED_FAILURE"
        quartodoc_skip_doctest = "quartodoc: +SKIP"

        chunker = lambda line: line.startswith((prompt, continuation))
        should_skip = (
            lambda line: quartodoc_skip_doctest in line or skip_doctest in line
        )

        for chunk in toolz.partitionby(chunker, lines):
            first, *rest = chunk

            # only attempt to execute or render code blocks that start with the
            # >>> prompt
            if first.startswith(prompt):
                # check whether to skip execution and if so, render the code
                # block as `python` (not `{python}`) if it's marked with
                # skip_doctest, expect_failure or quartodoc_skip_doctest
                if not any(map(should_skip, chunk)):
                    start, end = "{}"
                else:
                    start = end = ""

                result.append(f"```{start}python{end}")

                # if we expect failures, don't fail the notebook execution and
                # render the error message
                if expect_failure in first or any(
                    expect_failure in line for line in rest
                ):
                    assert (
                        start and end
                    ), "expected failure should never occur alongside a skipped doctest example"
                    result.append("#| error: true")

                # remove the quartodoc markers from the rendered code
                result.append(
                    first.replace(f"# {quartodoc_skip_doctest}", "")
                    .replace(quartodoc_skip_doctest, "")
                    .replace(f"# {expect_failure}", "")
                    .replace(expect_failure, "")
                )
                result.extend(rest)
                result.append("```\n")

        return "\n".join(result)
