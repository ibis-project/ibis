WITH t0 AS (
  SELECT
    t7.field_of_study AS field_of_study,
    UNNEST(
      CAST([{'years': '1970-71', 'degrees': t7."1970-71"}, {'years': '1975-76', 'degrees': t7."1975-76"}, {'years': '1980-81', 'degrees': t7."1980-81"}, {'years': '1985-86', 'degrees': t7."1985-86"}, {'years': '1990-91', 'degrees': t7."1990-91"}, {'years': '1995-96', 'degrees': t7."1995-96"}, {'years': '2000-01', 'degrees': t7."2000-01"}, {'years': '2005-06', 'degrees': t7."2005-06"}, {'years': '2010-11', 'degrees': t7."2010-11"}, {'years': '2011-12', 'degrees': t7."2011-12"}, {'years': '2012-13', 'degrees': t7."2012-13"}, {'years': '2013-14', 'degrees': t7."2013-14"}, {'years': '2014-15', 'degrees': t7."2014-15"}, {'years': '2015-16', 'degrees': t7."2015-16"}, {'years': '2016-17', 'degrees': t7."2016-17"}, {'years': '2017-18', 'degrees': t7."2017-18"}, {'years': '2018-19', 'degrees': t7."2018-19"}, {'years': '2019-20', 'degrees': t7."2019-20"}] AS STRUCT(years TEXT, degrees BIGINT)[])
    ) AS __pivoted__
  FROM humanities AS t7
), t1 AS (
  SELECT
    t0.field_of_study AS field_of_study,
    STRUCT_EXTRACT(t0.__pivoted__, 'years') AS years,
    STRUCT_EXTRACT(t0.__pivoted__, 'degrees') AS degrees
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
    t4.diff < CAST(0 AS TINYINT)
), anon_2 AS (
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
    anon_2.field_of_study AS field_of_study,
    anon_2.diff AS diff
  FROM anon_2
) AS t6