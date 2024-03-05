from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from public import public

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.expr.types.core import Expr
from ibis.expr.types.relations import Table
from ibis.expr.types.generic import Column, Scalar, Value, UnknownColumn

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class MatchRecognizePartitionBy(Value):
    pass


@public
class MatchRecognizeOrderBy(Value):
    pass


@public
class MatchRecognizeVariable(Value):
    """Variables to define and measure patterns."""

    def __getattr__(self, key: str) -> ir.Expr:
        # TODO (mehmet): Is `self.op().table.to_expr()` really the right method here?
        field = getattr(self.op().table.to_expr(), key)
        return ops.MatchRecognizeVariableField(field=field, variable=self).to_expr()

    def define(self, definition: ir.Expr) -> ir.MatchRecognizeVariable:
        """Sets the pattern variable definition for matching data rows.

        The `definition` should be an expression built with pattern
        variables, including this variable. A pattern variable
        essentially represents the set of all data rows matching its
        definition.  Thus its definition should be a boolean
        expression that evaluates to true or false for every given
        data row. If the variable definition evaluates to true for a
        given data row, then the data row gets mapped to the
        variable. A single data row can get mapped to multiple
        variables. Note that variables that are defined without an
        expression are always assigned to all the data rows. See the
        examples below for how to construct and define pattern
        variables.

        Parameters
        ----------
        definition
            The variable definition.

        Returns
        -------
        Variable
            Match-recognize variable with the definition.

        Examples
        --------
        Given an Ibis table `table`, create two pattern variables

        > var_a = table.pattern_variable("a")
        > var_b = table.pattern_variable("b")

        Then `var_a` can be defined as
        > import ibis
        > var_a.define(ibis.and_(var_a.int_col >= 0, var_b.bool_col))

        which implies that `var_a` gets assigned to data rows with
        non-negative `int_col` column value, and only if the last data
        row mapped to `var_b` has its `bool_col` column evaluating to
        true.  Note that the "last row" here is relative to the row
        that is being processed by the match-recognize operator.

        Logical offsets can be used to navigate within the set of data
        rows matched to a given pattern variable. For instance, the
        definition above can be modified as

        > var_a.define(ibis.and_(var_a.int_col >= 0, var_b.bool_col.last(2)))

        to base the mapping decision on the `bool_col` column of the
        second-last row that got mapped to `var_b` (until the current
        row being processed). Similarly, the second-first row can be
        used for the mapping decision as follows

        > var_a.define(ibis.and_(var_a.int_col >= 0, var_b.bool_col.first(2)))
        """
        # TODO (mehmet): `definition` should be a boolean expression.
        # How can we check for that?
        # E.g.,
        # if not definition.is_boolean():
        #     raise exc.IbisInputError(
        #         "Match recognize variable can be defined only with "
        #         "a boolean expression."
        #     )

        op = self.op()
        # TODO (mehmet): Constructing a new `MatchRecognizeVariable`
        # on top of another one leaves "redundant" nodes behind in
        # the expression tree. Is there a way to trim these redundant
        # nodes? Same goes for attaching the quantifier in `quantify()`.
        # Perhaps, bind + dereferencing logic should be used here,
        # but I cannot tell how.
        #
        # Calling `define()` and `quantify()` multiple times on the same
        # variable with new `MatchRecognizeVariableField` in the middle
        # can lead to many redundant nodes -- even though this should
        # not be a common use case. This is a direct consequence of
        # continuously building ops on top of each other instead of
        # keeping a single instance and "rebasing" others on it.
        # In my understanding, dereferencing logic works with table
        # expressions only, and that is why, perhaps, we should change
        # match-recognize ops from being `Value` to `Table` expressions.

        # TODO (mehmet): Should we check if `definition` contains this
        # variable, or leave that to user?

        return ops.MatchRecognizeVariable(
            name=op.name,
            table=op.table,
            definition=definition,
            quantifier=op.quantifier,
        ).to_expr()

    def quantify(
        self,
        min_num_rows: int = 0,
        max_num_rows: int = None,
        reluctant: bool = False,
    ):
        """Bounds the number of data rows that can be mapped to this variable.

        `min_num_rows` and `max_num_rows` determine the lower and
        upper bounds for the number of data rows that can be mapped to
        the pattern variable.

        Parameters
        ----------
        min_num_rows
            Minimum number of data rows that can be mapped to this variable.
            Set to `0` by default.
        max_num_rows
            Maximum number of data rows that can be mapped to this variable.
            When undefined (i.e. set as `None`), unlimited number data rows
            can get mapped to this variable.
        reluctant
            Whether to add reluctant flag for the quantifier. Each
            quantifier can be either greedy (default behavior) or
            reluctant. Greedy quantifiers try to match as many data
            rows as possible within the pattern while reluctant
            quantifiers try to match as few as possible.

        Returns
        -------
        Variable
            Match-recognize variable with the quantifier.

        Examples
        --------
        Given an Ibis table `table`, create a match-recognize variable

        > var_a = table.pattern_variable("a")

        By default, `var_a` can be mapped to any number data rows,
        including zero. The limits can be quantified as

        > var_a.quantify(0, 1)

        With this quantifier now, `var_a` can be mapped to either a
        single data row or none.
        """
        op = self.op()
        quantifier = ops.Quantifier(
            min_num_rows=min_num_rows,
            max_num_rows=max_num_rows,
            reluctant=reluctant,
        )

        return ops.MatchRecognizeVariable(
            name=op.name,
            table=op.table,
            definition=op.definition,
            quantifier=quantifier,
        ).to_expr()


@public
class MatchRecognizeVariableField(Value):
    def __getattr__(self, key: str) -> ir.Expr:
        return getattr(self.op().field.to_expr(), key)

    # TODO (mehmet): Adding all the aggregation functions here looks
    # like repeating what is already defined for the existing `Value`
    # expressions, e.g., those defined in ibis.expr.types.numeric.py.
    # Ideally, we would not need to define a separate op for
    # `MatchRecognizeVariableField` and just rely on the fields based on
    # the original table (from which the pattern variable has been
    # created).  With this however, I could not find a way to compile
    # table fields with the name of the pattern variables instead of the
    # table name, e.g. we want `a`.`int_col` instead of
    # `table`.`int_col` as the pattern variable is defined with name
    # "a".

    def mean(self):
        return ops.Mean(self).to_expr()

    def sum(self):
        return ops.Sum(self).to_expr()

    def min(self):
        return ops.Min(self).to_expr()

    def max(self):
        return ops.Max(self).to_expr()

    def first(self, offset: int = None):
        if offset and offset < 1:
            raise exc.IbisInputError("`offset` must be at least 1")

        return ops.First(self, offset).to_expr()

    def last(self, offset: int = None):
        if offset and offset < 1:
            raise exc.IbisInputError("`offset` must be at least 1")

        return ops.Last(self, offset).to_expr()


@public
class MatchRecognizeDefine(Value):
    pass


@public
class MatchRecognizePattern(Value):
    pass


@public
class MatchRecognizeMeasure(Value):
    pass


@public
class MatchRecognizeMeasures(Value):
    pass


@public
class MatchRecognizeAfterMatch(Value):
    pass


@public
class MatchRecognizeTable(Table):
    """Table expression for pattern matching."""
    pass


@public
def pattern_measurement(
    name: str,
    definition: ir.Expr,
) -> MatchRecognizeMeasure:
    """Creates a measurement for the match-recognize statement to output.

    Measurement `definition` is a numerical expression that must be
    constructed in terms of one or more pattern variables.
    Measurements define the outputs of the match-recognize statement.
    In this sense, `measures` clause can be thought as the `select`
    for `MATCH RECOGNIZE` statement.

    Measurements can be collected from only a subset of the data rows
    that get mapped to the variable by setting the logical offsets for
    the columns (via `.first()` and `.last()`). Similarly, aggregate
    values for the selected columns can be defined using the
    aggregation functions such as `.sum()`, `.mean()` etc.

    Parameters
    ----------
    name
        Name of the measurement
    definition
        Definition of the measurement

    Returns
    -------
    Measurement
        Expression representing the measurement.

    Examples
    --------
    Given an Ibis table `table`, create two pattern variables

    > var_a = table.pattern_variable("a")
    > var_b = table.pattern_variable("b")

    Then, the values of the column `int_col` of the data rows that
    get mapped to `var_a` can be measured as

    > import ibis
    > ibis.pattern_measurement("a_int_col", var_a.int_col)

    It is possible to define the measurement only for the last data
    row that gets mapped to `var_a` within the matched patterns as
    follows

    > ibis.pattern_measurement("a_int_col", var_a.int_col.last())

    Or, define it for the second-last data row as

    > ibis.pattern_measurement("a_int_col", var_a.int_col.last(2))

    It is also possible to define aggregated measurements as

    > ibis.pattern_measurement("a_int_col_mean", var_a.int_col.mean())
    > ibis.pattern_measurement("b_int_col_sum", var_b.int_col.sum())

    Measurements can be defined in terms of multiple variables as

    > ibis.pattern_measurement("a_minus_b", var_a.int_col - var_b.int_col)
    """

    return ops.MatchRecognizeMeasure(
        name=name,
        definition=definition,
    ).to_expr()


@public
def pattern_after_match(
    strategy: str,
    variable: ir.MatchRecognizeVariable = None,
) -> MatchRecognizeAfterMatch:
    """Creates an after-match strategy for pattern matching.

    The after-match strategy specifies where to start a new
    matching procedure after a complete match was found for a
    given pattern. There are currently four different strategies
    defined
    * "skip past last": Resumes the pattern matching at the next
        row after the last row of the current match.
    * "skip to next": Continues searching for a new match starting
        at the next row after the starting row of the match.
    * <"skip to last", variable>: Resumes the pattern matching
        at the last row that is mapped to `variable`.
    * <"skip to first", variable>: Resumes the pattern matching
        at the first row that is mapped to `variable`.

    Note that not all these strategies might be supported for
    the selected backend. Note also that the "skip to last"
    and "skip to first` require the pattern `variable` to be
    specified.

    Parameters
    ----------
    strategy
        After-match strategy in string. Four different possible
        values are as given above.
    variable
        Pattern variable to be included in the after-match
        expression.

    Returns
    -------
    After-match
        Expression representing the after-match strategy.

    Examples
    --------
    After-match strategies that do not relate to a pattern variable
    are created as
    > ibis.pattern_after_match("skip past last")
    > ibis.pattern_after_match("skip past last")

    Strategies that relate to a pattern variable, say `var_a`,
    are created as
    > ibis.pattern_after_match("skip past first", var_a)
    > ibis.pattern_after_match("skip past last", var_a)
    """

    from ibis.expr.operations.match_recognize import AfterMatchStrategy

    after_match_strategy = AfterMatchStrategy.from_str(strategy)
    return ops.MatchRecognizeAfterMatch(
        strategy=after_match_strategy,
        variable=variable,
    ).to_expr()
