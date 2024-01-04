"""APIs for creating user-defined functions."""

from __future__ import annotations

import ibis.legacy.udf.vectorized


class udf:
    @staticmethod
    def elementwise(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.elementwise."""

        return ibis.legacy.udf.vectorized.elementwise(input_type, output_type)

    @staticmethod
    def reduction(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.reduction."""
        return ibis.legacy.udf.vectorized.reduction(input_type, output_type)

    @staticmethod
    def analytic(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.analytic."""
        return ibis.legacy.udf.vectorized.analytic(input_type, output_type)
