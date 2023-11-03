WITH t0 AS (
  SELECT
    t7.field_of_study AS field_of_study,
    CAST(ROW(anon_2.years, anon_2.degrees) AS ROW(years VARCHAR, degrees BIGINT)) AS __pivoted__
  FROM humanities AS t7
  JOIN UNNEST(ARRAY[CAST(ROW('1970-71', t7."1970-71") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1975-76', t7."1975-76") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1980-81', t7."1980-81") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1985-86', t7."1985-86") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1990-91', t7."1990-91") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1995-96', t7."1995-96") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2000-01', t7."2000-01") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2005-06', t7."2005-06") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2010-11', t7."2010-11") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2011-12', t7."2011-12") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2012-13', t7."2012-13") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2013-14', t7."2013-14") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2014-15', t7."2014-15") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2015-16', t7."2015-16") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2016-17', t7."2016-17") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2017-18', t7."2017-18") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2018-19', t7."2018-19") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2019-20', t7."2019-20") AS ROW(years VARCHAR, degrees BIGINT))]) AS anon_2(years, degrees)
    ON TRUE
), t1 AS (
  SELECT
    t0.field_of_study AS field_of_study,
    t0.__pivoted__.years AS years,
    t0.__pivoted__.degrees AS degrees
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
    ARBITRARY(t3.diff) AS diff
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