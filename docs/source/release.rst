=============
Release Notes
=============

.. note::

   These release notes are for versions of ibis **1.0 and later**. Release
   notes for pre-1.0 versions of ibis can be found at :doc:`/release-pre-1.0`

* :bug:`2089` Pin "clickhouse-driver" to ">=0.1.3"
* :release:`1.3.0 <pending>`
* :support:`2077` Change omniscidb image tag from v5.0.0 to v5.1.0 on docker-compose recipe
* :feature:`2071` Improve many arguments UDF performance in pandas backend.
* :support:`2075` Disable Postgres tests on Windows CI.
* :feature:`2048` Introduce a top level vectorized UDF module (experimental). Implement element-wise UDF for pandas and PySpark backend.
* :bug:`2069` Fix load data stage for Linux CI
* :support:`2066` SUPP: Add support to Python 3.8
* :support:`2075` Disable Postgres tests on Windows CI.
* :feature:`2048` Introduce a top level vectorized UDF module (experimental). Implement element-wise UDF for pandas and PySpark backend.
* :bug:`2057` Fix datamgr.py fail if IBIS_TEST_OMNISCIDB_DATABASE=omnisci
* :bug:`2061` CI: Fix CI builds related to new pandas 1.0 compatibility
* :feature:`1976` Add DenseRank, RowNumber, MinRank, Count, PercentRank/CumeDist window operations to OmniSciDB
* :bug:`2055` Fix "cudf" import on OmniSciDB backend
* :feature:`2052` added possibility to run tests for separate backend via `make test BACKENDS=[YOUR BACKEND]`
* :bug:`2056` Fix data map for int8 on OmniSciDB backend
* :support:`2034` Add initial documentation for OmniSciDB, MySQL, PySpark and SparkSQL backends, add initial documentation for geospatial methods and add links to Ibis wiki page
* :bug:`2050` CI: Drop table only if it exists
* :feature:`2044` Implement covariance for bigquery backend
* :feature:`2035` Add support for  multi arguments window UDAF for the pandas backend
* :bug:`2041` Change pymapd connection parameter from "session_id" to "sessionid"
* :support:`2037` Drop support for Python 3.5
* :bug:`2023` HTML escape column names and types in png repr.
* :support:`2031` Change omniscidb image tag from v4.7.0 to v5.0.0 on docker-compose recipe
* :bug:`2030` Pin "semantic_version" to "<2.7" in the docs build CI, fix "builddoc" and "doc" section inside "Makefile" and skip mysql tzinfo on CI to allow to run MySQL using docker container on a hard disk drive.
* :bug:`2009` Fix pandas backend to treat trailing_window preceding arg as window bound rather than window size (e.g. preceding=0 now indicates current row rather than window size 0)
* :release:`1.2.0 <2019-06-24>`
* :feature:`1836` Add new geospatial functions to OmniSciDB backend
* :support:`1847` Skip SQLAlchemy backend tests in connect method in backends.py
* :bug:`1855 major` Fix call to psql causing failing CI
* :bug:`1851 major` Fix nested array literal repr
* :support:`1848` Validate order_by when using rows_with_max_lookback window
* :bug:`1850 major` Fix repr of empty schema
* :support:`1845` Generate release notes from commits
* :support:`1844` Raise exception on backends where rows_with_max_lookback can't be implemented
* :bug:`1843 major` Add max_lookback to window replace and combine functions
* :bug:`1837 major` Partially revert #1758
* :support:`1840` Tighter version spec for pytest
* :feature:`1838` allow pandas timedelta in rows_with_max_lookback
* :feature:`1825` Accept rows-with-max-lookback as preceding parameter
* :feature:`1787` PostGIS support
* :support:`1826` Allow passing a branch to ci/feedstock.py
* :support:`-` Bugs go into feature releases
* :support:`-` No space after :release:
* :release:`1.1.0 <2019-06-09>`
* :bug:`1819 major` Fix group_concat test and implementations
* :support:`1820` Remove decorator hacks and add custom markers
* :bug:`1818 major` Fix failing strftime tests on Python 3.7
* :bug:`1757 major` Remove unnecessary (and erroneous in some cases) frame clauses
* :support:`1814` Add development deps to setup.py
* :feature:`1809` Conslidate trailing window functions
* :bug:`1799 major` Chained mutate operations are buggy
* :support:`1805` Fix design and developer docs
* :support:`1810` Pin sphinx version to 2.0.1
* :feature:`1766` Call to_interval when casting integers to intervals
* :bug:`1783 major` Allow projections from joins to attempt fusion
* :feature:`1796` Add session feature to mapd client API
* :bug:`1798 major` Fix Python 3.5 dependency versions
* :feature:`1792` Add min periods parameter to Window
* :support:`1793` Add pep8speaks integration
* :support:`1821` Fix typo in UDF signature specification
* :feature:`1785` Allow strings for types in pandas UDFs
* :feature:`1790` Add missing date operations and struct field operation for the pandas backend
* :bug:`1789 major` Fix compatibility and bugs associated with pandas toposort reimplementation
* :bug:`1772 major` Fix outer_join generating LEFT join instead of FULL OUTER
* :feature:`1771` Add window operations to the OmniSci backend
* :feature:`1758` Reimplement the pandas backend using topological sort
* :support:`1779` Clean up most xpassing tests
* :bug:`1782 major` NullIf should enforce that its arguments are castable to a common type
* :support:`1781` Update omnisci container version
* :feature:`1778` Add marker for xfailing specific backends
* :feature:`1777` Enable window function tests where possible
* :bug:`1775 major` Fix conda create command in documentation
* :support:`1776` Constrain PyMapD version to get passing builds
* :bug:`1765 major` Fix preceding and following with ``None``
* :support:`1763` Remove warnings and clean up some docstrings
* :support:`1638` Add StringToTimestamp as unsupported
* :feature:`1743` is_computable_arg dispatcher
* :support:`1759` Add isort pre-commit hooks
* :feature:`1753` Added float32 and geospatial types for create table from schema
* :bug:`1661 major` PostgreSQL interval type not recognized
* :support:`1750` Add Python 3.5 testing back to CI
* :support:`1700` Re-enable CI for building step
* :support:`1749` Update README reference to MapD to say OmniSci
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
