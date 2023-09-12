from __future__ import annotations

from io import StringIO


class _CaseFormatter:
    def __init__(self, translator, base, cases, results, default):
        self.translator = translator
        self.base = base
        self.cases = cases
        self.results = results
        self.default = default

        # HACK
        self.indent = 2
        self.multiline = len(cases) > 1
        self.buf = StringIO()

    def get_result(self):
        self.buf.seek(0)

        self.buf.write("CASE")
        if self.base is not None:
            base_str = self.translator.translate(self.base)
            self.buf.write(f" {base_str}")

        for case, result in zip(self.cases, self.results):
            self._next_case()
            case_str = self.translator.translate(case)
            result_str = self.translator.translate(result)
            self.buf.write(f"WHEN {case_str} THEN {result_str}")

        if self.default is not None:
            self._next_case()
            default_str = self.translator.translate(self.default)
            self.buf.write(f"ELSE {default_str}")

        if self.multiline:
            self.buf.write("\nEND")
        else:
            self.buf.write(" END")

        return self.buf.getvalue()

    def _next_case(self):
        if self.multiline:
            self.buf.write("\n{}".format(" " * self.indent))
        else:
            self.buf.write(" ")


def simple_case(translator, op):
    formatter = _CaseFormatter(translator, op.base, op.cases, op.results, op.default)
    return formatter.get_result()


def searched_case(translator, op):
    formatter = _CaseFormatter(translator, None, op.cases, op.results, op.default)
    return formatter.get_result()
