=============
Release Notes
=============

.. note::

   These release notes are for versions of ibis **1.0 and later**. Release
   notes for pre-1.0 versions of ibis can be found at :doc:`/release-pre-1.0`

* :release:`1.0.0 <2019-03-26>`
* :support:`1748` Do not build universal wheels
* :support:`1747` Remove tag prefix from versioneer
* :support:`1746` Use releases to manage documentation
* :feature:`1735` Add black as a pre-commit hook
* :feature:`1680` Add support for the arbitrary aggregate in the mapd backend
* :bug:`1745` Make ``dev/merge-pr.py`` script handle PR branches
* :feature:`1731` Add SQL method for the MapD backend
* :feature:`1744` Clean up merge PR script and use the actual merge feature of GitHub
* :bug:`1742` Fix ``NULLIF`` implementation for the pandas backend
* :bug:`1737` Fix casting to float in the MapD backend
* :bug:`1741` Fix testing for BigQuery after auth flow update
* :feature:`1723` Add cross join to the pandas backend
* :bug:`1738` Fix skipping for new BigQuery auth flow
* :bug:`1732` Fix bug in ``TableExpr.drop``
* :feature:`1727` Implement default handler for multiple client ``pre_execute``
* :feature:`1728` Implement BigQuery auth using ``pydata_google_auth``
* :bug:`1729` Filter the ``raw`` warning from newer pandas to support older pandas
* :bug:`1706` Fix BigQuery credentials link
* :feature:`1712` Timestamp literal accepts a timezone parameter
* :feature:`1725` Remove support for passing integers to ``ibis.timestamp``
* :feature:`1704` Add ``find_nodes`` to lineage
* :feature:`1714` Remove a bunch of deprecated APIs and clean up warnings
* :feature:`1716` Implement table distinct for the pandas backend
* :feature:`1678` Implement geospatial functions for MapD
* :feature:`1666` Implement geospatial types for MapD
* :support:`1694` Use cudf instead of pygdf
* :bug:`1639` Add Union as an unsuppoted operation for MapD
* :bug:`1705` Fix visualizing an ibis expression when showing a selection after a table join
* :bug:`1659` Fix MapD exception for ``toDateTime``
* :bug:`1701` Use ``==`` to compare strings
* :support:`1696` Fix multiple CI issues
* :feature:`1685` Add pre commit hook
* :support:`1681` Update mapd ci to v4.4.1
* :feature:`1686` Getting started with mapd, mysql and pandas
* :support:`1672` Enabled mysql CI on azure pipelines
* :support:`-` Update docs to reflect Apache Impala and Kudu as ASF TLPs
* :feature:`1675` Support column names with special characters in mapd
* :support:`1670` Remove support for Python 2
* :feature:`1669` Allow operations to hide arguments from display
* :bug:`1647` Resolves joining with different column names
* :bug:`1643` Fix map get with compatible types
* :feature:`1636` Remove implicit ordering requirements in the PostgreSQL backend
* :feature:`1655` Add cross join operator to MapD
* :support:`1667` Fix flake8 and many other warnings
* :bug:`1653` Fixed where operator for MapD
* :support:`1664` Update README.md for impala and kudu
* :support:`1660` Remove defaults as a channel from azure pipelines
* :support:`1658` Fixes a very typo in the pandas/core.py docstring
* :support:`1657` Unpin clickhouse-driver version
* :bug:`1648` Remove parameters from mapd
* :bug:`1651` Make sure we cast when NULL is else in CASE expressions
* :support:`1650` Add test for reduction returning lists
* :feature:`1637` Fix UDF bugs and add support for non-aggregate analytic functions
* :support:`1646` Fix Azure VM image name
* :support:`1641` Updated MapD server-CI
* :support:`1645` Add TableExpr.drop to API documentation
* :support:`1642` Fix Azure deployment step
* :support:`-` Update README.md
* :support:`1640` Set up CI with Azure Pipelines
* :feature:`1627` Support string slicing with other expressions
* :feature:`1618` Publish the ibis roadmap
* :feature:`1604` Implement ``approx_median`` in BigQuery
* :feature:`1611` Make ibis node instances hashable
* :bug:`1600` Fix equality
* :feature:`1608` Add ``range_window`` and ``trailing_range_window`` to docs
* :support:`1609` Fix conda builds
* :release:`0.14.0 <2018-08-23>`
