WITH t0 AS (
  SELECT
    t6.field_of_study AS field_of_study,
    UNNEST(
      CAST(LIST_VALUE(
        '1970-71',
        '1975-76',
        '1980-81',
        '1985-86',
        '1990-91',
        '1995-96',
        '2000-01',
        '2005-06',
        '2010-11',
        '2011-12',
        '2012-13',
        '2013-14',
        '2014-15',
        '2015-16',
        '2016-17',
        '2017-18',
        '2018-19',
        '2019-20'
      ) AS TEXT[])
    ) AS years,
    UNNEST(
      CAST(LIST_VALUE(
        t6."1970-71",
        t6."1975-76",
        t6."1980-81",
        t6."1985-86",
        t6."1990-91",
        t6."1995-96",
        t6."2000-01",
        t6."2005-06",
        t6."2010-11",
        t6."2011-12",
        t6."2012-13",
        t6."2013-14",
        t6."2014-15",
        t6."2015-16",
        t6."2016-17",
        t6."2017-18",
        t6."2018-19",
        t6."2019-20"
      ) AS BIGINT[])
    ) AS degrees
  FROM humanities AS t6
), t1 AS (
  SELECT
    t0.field_of_study AS field_of_study,
    t0.years AS years,
    t0.degrees AS degrees,
    FIRST_VALUE(t0.degrees) OVER (PARTITION BY t0.field_of_study ORDER BY t0.years ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS earliest_degrees,
    LAST_VALUE(t0.degrees) OVER (PARTITION BY t0.field_of_study ORDER BY t0.years ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS latest_degrees
  FROM t0
), t2 AS (
  SELECT
    t1.field_of_study AS field_of_study,
    t1.years AS years,
    t1.degrees AS degrees,
    t1.earliest_degrees AS earliest_degrees,
    t1.latest_degrees AS latest_degrees,
    t1.latest_degrees - t1.earliest_degrees AS diff
  FROM t1
), t3 AS (
  SELECT
    t2.field_of_study AS field_of_study,
    FIRST(t2.diff) AS diff
  FROM t2
  GROUP BY
    1
), anon_1 AS (
  SELECT
    t3.field_of_study AS field_of_study,
    t3.diff AS diff
  FROM t3
  ORDER BY
    t3.diff DESC
  LIMIT 10
), t4 AS (
  SELECT
    t3.field_of_study AS field_of_study,
    t3.diff AS diff
  FROM t3
  WHERE
    t3.diff < CAST(0 AS SMALLINT)
), anon_2 AS (
  SELECT
    t4.field_of_study AS field_of_study,
    t4.diff AS diff
  FROM t4
  ORDER BY
    t4.diff
  LIMIT 10
)
SELECT
  t5.field_of_study,
  t5.diff
FROM (
  SELECT
    anon_1.field_of_study AS field_of_study,
    anon_1.diff AS diff
  FROM anon_1
  UNION ALL
  SELECT
    anon_2.field_of_study AS field_of_study,
    anon_2.diff AS diff
  FROM anon_2
) AS t5