from __future__ import annotations

import sqlglot.expressions as sge


# TODO (mehmet): Can we eliminate the classes here and use the
# readily available sqlglot expressions?

class MatchRecognizePartitionBy(sge.Expression):
    arg_types = {
        "columns": True,
    }


class MatchRecognizeOrderBy(sge.Expression):
    arg_types = {
        "columns": True,
    }


class TableSymbol(sge.Expression):
    arg_types = {
        "this": True,
    }


class Quantifier(sge.Expression):
    arg_types = {
        "min_num_rows": True,
        "max_num_rows": False,
        "reluctant": False,
    }


class MatchRecognizeVariable(sge.Expression):
    arg_types = {
        "name": True,
        "definition": False,
        "quantifier": False,
    }


class MatchRecognizeVariableField(sge.Expression):
    arg_types = {
        "symbol": True,
    }


class MatchRecognizeDefine(sge.Expression):
    arg_types = {
        "variables": True,
    }


class MatchRecognizePattern(sge.Expression):
    arg_types = {
        "variables": True,
    }


class MatchRecognizeMeasure(sge.Expression):
    arg_types = {
        "name": True,
        "definition": True,
    }


class MatchRecognizeMeasures(sge.Expression):
    arg_types = {
        "measures": True,
    }

# TODO (mehmet): Can we replace `MatchRecognizeAfterMatch` and `MatchRecognizeOutputMode`
# with an existing sqlglot expression.
class MatchRecognizeAfterMatch(sge.Expression):
    arg_types = {
        "this": True,
    }


class MatchRecognizeOutputMode(sge.Expression):
    arg_types = {
        "this": True,
    }


class MatchRecognize(sge.Expression):
    arg_types = {
        "partition_by": False,
        "order_by": False,
        "define": True,
        "pattern": True,
        "after_match": True,
        "measures": True,
        "output_mode": True,
        # "alias": False,
    }


class MatchRecognizeTable(sge.Expression):
    arg_types = {
        "table": True,
        "match_recognize": True,
    }
