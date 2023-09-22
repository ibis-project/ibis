WITH t0 AS (
  SELECT
    t7.field_of_study AS field_of_study,
    ROW(anon_2.f1, anon_2.f2) AS __pivoted__
  FROM humanities AS t7
  JOIN UNNEST(ARRAY[ROW(CAST('1970-71' AS TEXT), CAST(t7."1970-71" AS BIGINT)), ROW(CAST('1975-76' AS TEXT), CAST(t7."1975-76" AS BIGINT)), ROW(CAST('1980-81' AS TEXT), CAST(t7."1980-81" AS BIGINT)), ROW(CAST('1985-86' AS TEXT), CAST(t7."1985-86" AS BIGINT)), ROW(CAST('1990-91' AS TEXT), CAST(t7."1990-91" AS BIGINT)), ROW(CAST('1995-96' AS TEXT), CAST(t7."1995-96" AS BIGINT)), ROW(CAST('2000-01' AS TEXT), CAST(t7."2000-01" AS BIGINT)), ROW(CAST('2005-06' AS TEXT), CAST(t7."2005-06" AS BIGINT)), ROW(CAST('2010-11' AS TEXT), CAST(t7."2010-11" AS BIGINT)), ROW(CAST('2011-12' AS TEXT), CAST(t7."2011-12" AS BIGINT)), ROW(CAST('2012-13' AS TEXT), CAST(t7."2012-13" AS BIGINT)), ROW(CAST('2013-14' AS TEXT), CAST(t7."2013-14" AS BIGINT)), ROW(CAST('2014-15' AS TEXT), CAST(t7."2014-15" AS BIGINT)), ROW(CAST('2015-16' AS TEXT), CAST(t7."2015-16" AS BIGINT)), ROW(CAST('2016-17' AS TEXT), CAST(t7."2016-17" AS BIGINT)), ROW(CAST('2017-18' AS TEXT), CAST(t7."2017-18" AS BIGINT)), ROW(CAST('2018-19' AS TEXT), CAST(t7."2018-19" AS BIGINT)), ROW(CAST('2019-20' AS TEXT), CAST(t7."2019-20" AS BIGINT))]) AS anon_2(f1 TEXT, f2 BIGINT)
    ON TRUE
), t1 AS (
  SELECT
    t0.field_of_study AS field_of_study,
    (
      t0.__pivoted__
    ).f1 AS years,
    (
      t0.__pivoted__
    ).f2 AS degrees
  FROM t0
), t2 AS (
  SELECT
    t1.field_of_study AS field_of_study,
    t1.years AS years,
    t1.degrees AS degrees,
    FIRST_VALUE(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS earliest_degrees,
    LAST_VALUE(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS latest_degrees
  FROM t1
), t3 AS (
  SELECT
    t2.field_of_study AS field_of_study,
    t2.years AS years,
    t2.degrees AS degrees,
    t2.earliest_degrees AS earliest_degrees,
    t2.latest_degrees AS latest_degrees,
    t2.latest_degrees - t2.earliest_degrees AS diff
  FROM t2
), t4 AS (
  SELECT
    t3.field_of_study AS field_of_study,
    FIRST(t3.diff) AS diff
  FROM t3
  GROUP BY
    1
), anon_1 AS (
  SELECT
    t4.field_of_study AS field_of_study,
    t4.diff AS diff
  FROM t4
  ORDER BY
    t4.diff DESC
  LIMIT 10
), t5 AS (
  SELECT
    t4.field_of_study AS field_of_study,
    t4.diff AS diff
  FROM t4
  WHERE
    t4.diff < 0
), anon_3 AS (
  SELECT
    t5.field_of_study AS field_of_study,
    t5.diff AS diff
  FROM t5
  ORDER BY
    t5.diff ASC
  LIMIT 10
)
SELECT
  t6.field_of_study,
  t6.diff
FROM (
  SELECT
    anon_1.field_of_study AS field_of_study,
    anon_1.diff AS diff
  FROM anon_1
  UNION ALL
  SELECT
    anon_3.field_of_study AS field_of_study,
    anon_3.diff AS diff
  FROM anon_3
) AS t6