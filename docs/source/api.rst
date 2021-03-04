.. currentmodule:: ibis
.. _api:

*************
API Reference
*************

Top-level expression APIs
-------------------------

.. currentmodule:: ibis

These methods are available directly in the ``ibis`` module namespace.

.. autosummary::
   :toctree: generated/

   case
   literal
   schema
   table
   timestamp
   where
   ifelse
   coalesce
   greatest
   least
   negate
   desc
   now
   NA
   null
   expr_list
   row_number
   window
   range_window
   trailing_window
   cumulative_window
   trailing_range_window
   random

.. _api.expr:

General expression methods
--------------------------

.. currentmodule:: ibis.expr.api

.. autosummary::
   :toctree: generated/

   Expr.compile
   Expr.equals
   Expr.execute
   Expr.pipe
   Expr.verify

.. _api.table:

Table methods
-------------

.. currentmodule:: ibis.expr.api

.. autosummary::
   :toctree: generated/

   TableExpr.aggregate
   TableExpr.asof_join
   TableExpr.count
   TableExpr.distinct
   TableExpr.drop
   TableExpr.info
   TableExpr.filter
   TableExpr.get_column
   TableExpr.get_columns
   TableExpr.group_by
   TableExpr.groupby
   TableExpr.limit
   TableExpr.mutate
   TableExpr.projection
   TableExpr.relabel
   TableExpr.rowid
   TableExpr.schema
   TableExpr.set_column
   TableExpr.sort_by
   TableExpr.union
   TableExpr.view

   TableExpr.join
   TableExpr.cross_join
   TableExpr.inner_join
   TableExpr.left_join
   TableExpr.outer_join
   TableExpr.semi_join
   TableExpr.anti_join


Grouped table methods
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   GroupedTableExpr.aggregate
   GroupedTableExpr.count
   GroupedTableExpr.having
   GroupedTableExpr.mutate
   GroupedTableExpr.order_by
   GroupedTableExpr.over
   GroupedTableExpr.projection
   GroupedTableExpr.size

Generic value methods
---------------------

.. _api.functions:

Scalar or column methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ValueExpr.between
   ValueExpr.cast
   ValueExpr.coalesce
   ValueExpr.fillna
   ValueExpr.isin
   ValueExpr.notin
   ValueExpr.nullif
   ValueExpr.hash
   ValueExpr.isnull
   ValueExpr.notnull
   ValueExpr.over
   ValueExpr.typeof

   ValueExpr.case
   ValueExpr.cases
   ValueExpr.substitute

Column methods
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ColumnExpr.distinct

   ColumnExpr.count
   ColumnExpr.min
   ColumnExpr.max
   ColumnExpr.approx_median
   ColumnExpr.approx_nunique
   ColumnExpr.group_concat
   ColumnExpr.nunique
   ColumnExpr.summary

   ColumnExpr.value_counts

   ColumnExpr.first
   ColumnExpr.last
   ColumnExpr.dense_rank
   ColumnExpr.rank
   ColumnExpr.lag
   ColumnExpr.lead
   ColumnExpr.cummin
   ColumnExpr.cummax

General numeric methods
-----------------------

Scalar or column methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   NumericValue.abs
   NumericValue.ceil
   NumericValue.floor
   NumericValue.sign
   NumericValue.exp
   NumericValue.sqrt
   NumericValue.log
   NumericValue.ln
   NumericValue.log2
   NumericValue.log10
   NumericValue.round
   NumericValue.nullifzero
   NumericValue.zeroifnull
   NumericValue.add
   NumericValue.sub
   NumericValue.mul
   NumericValue.div
   NumericValue.pow
   NumericValue.rdiv
   NumericValue.rsub



Column methods
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   NumericColumn.sum
   NumericColumn.mean

   NumericColumn.std
   NumericColumn.var

   NumericColumn.cumsum
   NumericColumn.cummean

   NumericColumn.bottomk
   NumericColumn.topk
   NumericColumn.bucket
   NumericColumn.histogram

Integer methods
---------------

Scalar or column methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   IntegerValue.convert_base
   IntegerValue.to_timestamp
   IntegerColumn.bit_and
   IntegerColumn.bit_or
   IntegerColumn.bit_xor

.. _api.string:

String methods
--------------

All string operations are valid either on scalar or array values

.. autosummary::
   :toctree: generated/

   StringValue.convert_base
   StringValue.length
   StringValue.lower
   StringValue.upper
   StringValue.reverse
   StringValue.ascii_str
   StringValue.strip
   StringValue.lstrip
   StringValue.rstrip
   StringValue.capitalize
   StringValue.contains
   StringValue.like
   StringValue.to_timestamp
   StringValue.parse_url
   StringValue.substr
   StringValue.left
   StringValue.right
   StringValue.repeat
   StringValue.find
   StringValue.translate
   StringValue.find_in_set
   StringValue.join
   StringValue.replace
   StringValue.lpad
   StringValue.rpad

   StringValue.rlike
   StringValue.re_search
   StringValue.re_extract
   StringValue.re_replace

.. _api.timestamp:

Timestamp methods
-----------------

All timestamp operations are valid either on scalar or array values

.. autosummary::
   :toctree: generated/

   TimestampValue.strftime
   TimestampValue.year
   TimestampValue.month
   TimestampValue.day
   TimestampValue.day_of_week
   TimestampValue.epoch_seconds
   TimestampValue.hour
   TimestampValue.minute
   TimestampValue.second
   TimestampValue.millisecond
   TimestampValue.truncate
   TimestampValue.time
   TimestampValue.date
   TimestampValue.add
   TimestampValue.radd
   TimestampValue.sub
   TimestampValue.rsub

.. _api.date:

Date methods
------------

.. autosummary::
   :toctree: generated/

   DateValue.strftime
   DateValue.year
   DateValue.month
   DateValue.day
   DateValue.day_of_week
   DateValue.epoch_seconds
   DateValue.truncate
   DateValue.add
   DateValue.radd
   DateValue.sub
   DateValue.rsub

.. _api.dow:

Day of week methods
-------------------

.. currentmodule:: ibis.expr.types

.. autosummary::
   :toctree: generated/

   DayOfWeek.index
   DayOfWeek.full_name

.. currentmodule:: ibis.expr.api

.. _api.time:

Time methods
------------

.. autosummary::
   :toctree: generated/

   TimeValue.between
   TimeValue.truncate
   TimeValue.hour
   TimeValue.minute
   TimeValue.second
   TimeValue.millisecond
   TimeValue.add
   TimeValue.radd
   TimeValue.sub
   TimeValue.rsub

.. _api.interval:

Interval methods
----------------

.. autosummary::
   :toctree: generated/

   IntervalValue.to_unit
   IntervalValue.years
   IntervalValue.quarters
   IntervalValue.months
   IntervalValue.weeks
   IntervalValue.days
   IntervalValue.hours
   IntervalValue.minutes
   IntervalValue.seconds
   IntervalValue.milliseconds
   IntervalValue.microseconds
   IntervalValue.nanoseconds
   IntervalValue.add
   IntervalValue.radd
   IntervalValue.sub
   IntervalValue.mul
   IntervalValue.rmul
   IntervalValue.floordiv
   IntervalValue.negate


Boolean methods
---------------

.. autosummary::
   :toctree: generated/

   BooleanValue.ifelse


.. autosummary::
   :toctree: generated/

   BooleanColumn.any
   BooleanColumn.all
   BooleanColumn.cumany
   BooleanColumn.cumall

Category methods
----------------

Category is a logical type with either a known or unknown cardinality. Values
are represented semantically as integers starting at 0.

.. autosummary::
   :toctree: generated/

   CategoryValue.label

Decimal methods
---------------

.. autosummary::
   :toctree: generated/

   DecimalValue.precision
   DecimalValue.scale

.. _api.struct:

Struct methods
-----------------

Scalar or column methods
~~~~~~~~~~~~~~~~~~~~~~~~

Values in a ``StructValue`` can be accessed using indexing, e.g. ``struct_expr['my_col']``. See :meth:`StructValue.__getitem__`.

.. autosummary::
   :toctree: generated/

   StructValue.destructure
   StructValue.__getitem__

.. _api.geospatial:

Geospatial methods
-------------------

Scalar or column methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   GeoSpatialValue.area
   GeoSpatialValue.as_binary
   GeoSpatialValue.as_ewkb
   GeoSpatialValue.as_ewkt
   GeoSpatialValue.as_text
   GeoSpatialValue.azimuth
   GeoSpatialValue.buffer
   GeoSpatialValue.centroid
   GeoSpatialValue.contains
   GeoSpatialValue.contains_properly
   GeoSpatialValue.covers
   GeoSpatialValue.covered_by
   GeoSpatialValue.crosses
   GeoSpatialValue.d_fully_within
   GeoSpatialValue.d_within
   GeoSpatialValue.difference
   GeoSpatialValue.disjoint
   GeoSpatialValue.distance
   GeoSpatialValue.end_point
   GeoSpatialValue.envelope
   GeoSpatialValue.equals
   GeoSpatialValue.geometry_n
   GeoSpatialValue.geometry_type
   GeoSpatialValue.intersection
   GeoSpatialValue.intersects
   GeoSpatialValue.is_valid
   GeoSpatialValue.line_locate_point
   GeoSpatialValue.line_merge
   GeoSpatialValue.line_substring
   GeoSpatialValue.length
   GeoSpatialValue.max_distance
   GeoSpatialValue.n_points
   GeoSpatialValue.n_rings
   GeoSpatialValue.ordering_equals
   GeoSpatialValue.overlaps
   GeoSpatialValue.perimeter
   GeoSpatialValue.point_n
   GeoSpatialValue.set_srid
   GeoSpatialValue.simplify
   GeoSpatialValue.srid
   GeoSpatialValue.start_point
   GeoSpatialValue.touches
   GeoSpatialValue.transform
   GeoSpatialValue.union
   GeoSpatialValue.within
   GeoSpatialValue.x
   GeoSpatialValue.x_max
   GeoSpatialValue.x_min
   GeoSpatialValue.y
   GeoSpatialValue.y_max
   GeoSpatialValue.y_min

Column methods
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   GeoSpatialColumn.unary_union
