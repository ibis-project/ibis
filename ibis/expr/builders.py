import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir


class TypedCaseBuilder:
    __slots__ = ()

    def type(self):
        return rlz.highest_precedence_dtype(self.results)

    def else_(self, result_expr):
        """
        Specify

        Returns
        -------
        builder : CaseBuilder
        """
        kwargs = {
            slot: getattr(self, slot)
            for slot in self.__slots__
            if slot != 'default'
        }

        result_expr = rlz.any(result_expr)
        kwargs['default'] = result_expr
        # Maintain immutability
        return type(self)(**kwargs)

    def end(self):
        default = self.default
        if default is None:
            default = ir.null().cast(self.type())

        args = [
            getattr(self, slot) for slot in self.__slots__ if slot != 'default'
        ]
        args.append(default)
        op = self.__class__.case_op(*args)
        return op.to_expr()


class SimpleCaseBuilder(TypedCaseBuilder):
    __slots__ = 'base', 'cases', 'results', 'default'

    case_op = ops.SimpleCase

    def __init__(self, base, cases=None, results=None, default=None):
        self.base = base
        self.cases = list(cases if cases is not None else [])
        self.results = list(results if results is not None else [])
        self.default = default

    def when(self, case_expr, result_expr):
        """
        Add a new case-result pair.

        Parameters
        ----------
        case : Expr
          Expression to equality-compare with base expression. Must be
          comparable with the base.
        result : Expr
          Value when the case predicate evaluates to true.

        Returns
        -------
        builder : CaseBuilder
        """
        case_expr = rlz.any(case_expr)
        result_expr = rlz.any(result_expr)

        if not rlz.comparable(self.base, case_expr):
            raise TypeError(
                'Base expression and passed case are not ' 'comparable'
            )

        cases = list(self.cases)
        cases.append(case_expr)

        results = list(self.results)
        results.append(result_expr)

        # Maintain immutability
        return type(self)(self.base, cases, results, self.default)


class SearchedCaseBuilder(TypedCaseBuilder):
    __slots__ = 'cases', 'results', 'default'

    case_op = ops.SearchedCase

    def __init__(self, cases=None, results=None, default=None):
        self.cases = list(cases if cases is not None else [])
        self.results = list(results if results is not None else [])
        self.default = default

    def when(self, case_expr, result_expr):
        """
        Add a new case-result pair.

        Parameters
        ----------
        case : Expr
          Expression to equality-compare with base expression. Must be
          comparable with the base.
        result : Expr
          Value when the case predicate evaluates to true.

        Returns
        -------
        builder : CaseBuilder
        """
        case_expr = rlz.any(case_expr)
        result_expr = rlz.any(result_expr)

        if not isinstance(case_expr, ir.BooleanValue):
            raise TypeError(case_expr)

        cases = list(self.cases)
        cases.append(case_expr)

        results = list(self.results)
        results.append(result_expr)

        # Maintain immutability
        return type(self)(cases, results, self.default)
