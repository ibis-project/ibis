from __future__ import annotations

from abc import abstractmethod

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.common.grounds import Concrete


class Builder(Concrete):
    @property
    @abstractmethod
    def __type__(self):
        ...


class CaseBuilder(Builder):
    results = rlz.optional(rlz.tuple_of(rlz.any), default=[])
    default = rlz.optional(rlz.any)

    def type(self):
        return rlz.highest_precedence_dtype(self.results)

    def when(self, case_expr, result_expr):
        """Add a new case-result pair.

        Parameters
        ----------
        case_expr
            Expression to equality-compare with base expression. Must be
            comparable with the base.
        result_expr
            Value when the case predicate evaluates to true.
        """
        cases = self.cases + (case_expr,)
        results = self.results + (result_expr,)
        return self.copy(cases=cases, results=results)

    def else_(self, result_expr):
        """Construct an `ELSE` expression."""
        return self.copy(default=result_expr)

    def end(self):
        default = self.default
        if default is None:
            default = ir.null().cast(self.type())

        kwargs = dict(zip(self.__argnames__, self.__args__))
        kwargs["default"] = default

        return self.__type__(**kwargs).to_expr()


class SearchedCaseBuilder(CaseBuilder):
    __type__ = ops.SearchedCase
    cases = rlz.optional(rlz.tuple_of(rlz.boolean), default=[])


class SimpleCaseBuilder(CaseBuilder):
    __type__ = ops.SimpleCase
    base = rlz.any
    cases = rlz.optional(rlz.tuple_of(rlz.any), default=[])

    def when(self, case_expr, result_expr):
        """Add a new case-result pair.

        Parameters
        ----------
        case_expr
            Expression to equality-compare with base expression. Must be
            comparable with the base.
        result_expr
            Value when the case predicate evaluates to true.
        """
        case_expr = rlz.any(case_expr)
        if not rlz.comparable(self.base, case_expr):
            raise TypeError(
                f'Base expression {rlz._arg_type_error_format(self.base)} and '
                f'case {rlz._arg_type_error_format(case_expr)} are not comparable'
            )
        return super().when(case_expr, result_expr)
