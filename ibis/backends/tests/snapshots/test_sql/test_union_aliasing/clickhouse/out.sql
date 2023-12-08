SELECT
  t8.field_of_study AS field_of_study,
  t8.diff AS diff
FROM (
  SELECT
    t3.field_of_study AS field_of_study,
    t3.diff AS diff
  FROM (
    SELECT
      t2.field_of_study AS field_of_study,
      any(t2.diff) AS diff
    FROM (
      SELECT
        t1.field_of_study AS field_of_study,
        t1.years AS years,
        t1.degrees AS degrees,
        any(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS earliest_degrees,
        anyLast(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS latest_degrees,
        anyLast(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - any(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS diff
      FROM (
        SELECT
          t0.field_of_study AS field_of_study,
          CAST(arrayJoin(
            [CAST(tuple('1970-71', t0."1970-71") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1975-76', t0."1975-76") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1980-81', t0."1980-81") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1985-86', t0."1985-86") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1990-91', t0."1990-91") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1995-96', t0."1995-96") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2000-01', t0."2000-01") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2005-06', t0."2005-06") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2010-11', t0."2010-11") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2011-12', t0."2011-12") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2012-13', t0."2012-13") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2013-14', t0."2013-14") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2014-15', t0."2014-15") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2015-16', t0."2015-16") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2016-17', t0."2016-17") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2017-18', t0."2017-18") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2018-19', t0."2018-19") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2019-20', t0."2019-20") AS Tuple(years Nullable(String), degrees Nullable(Int64)))]
          ).1 AS Nullable(String)) AS years,
          CAST(arrayJoin(
            [CAST(tuple('1970-71', t0."1970-71") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1975-76', t0."1975-76") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1980-81', t0."1980-81") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1985-86', t0."1985-86") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1990-91', t0."1990-91") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1995-96', t0."1995-96") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2000-01', t0."2000-01") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2005-06', t0."2005-06") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2010-11', t0."2010-11") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2011-12', t0."2011-12") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2012-13', t0."2012-13") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2013-14', t0."2013-14") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2014-15', t0."2014-15") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2015-16', t0."2015-16") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2016-17', t0."2016-17") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2017-18', t0."2017-18") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2018-19', t0."2018-19") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2019-20', t0."2019-20") AS Tuple(years Nullable(String), degrees Nullable(Int64)))]
          ).2 AS Nullable(Int64)) AS degrees
        FROM humanities AS t0
      ) AS t1
    ) AS t2
    GROUP BY
      t2.field_of_study
  ) AS t3
  ORDER BY
    t3.diff DESC
  LIMIT 10
  UNION ALL
  SELECT
    t3.field_of_study AS field_of_study,
    t3.diff AS diff
  FROM (
    SELECT
      t2.field_of_study AS field_of_study,
      any(t2.diff) AS diff
    FROM (
      SELECT
        t1.field_of_study AS field_of_study,
        t1.years AS years,
        t1.degrees AS degrees,
        any(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS earliest_degrees,
        anyLast(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS latest_degrees,
        anyLast(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - any(t1.degrees) OVER (PARTITION BY t1.field_of_study ORDER BY t1.years ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS diff
      FROM (
        SELECT
          t0.field_of_study AS field_of_study,
          CAST(arrayJoin(
            [CAST(tuple('1970-71', t0."1970-71") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1975-76', t0."1975-76") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1980-81', t0."1980-81") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1985-86', t0."1985-86") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1990-91', t0."1990-91") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1995-96', t0."1995-96") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2000-01', t0."2000-01") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2005-06', t0."2005-06") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2010-11', t0."2010-11") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2011-12', t0."2011-12") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2012-13', t0."2012-13") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2013-14', t0."2013-14") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2014-15', t0."2014-15") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2015-16', t0."2015-16") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2016-17', t0."2016-17") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2017-18', t0."2017-18") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2018-19', t0."2018-19") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2019-20', t0."2019-20") AS Tuple(years Nullable(String), degrees Nullable(Int64)))]
          ).1 AS Nullable(String)) AS years,
          CAST(arrayJoin(
            [CAST(tuple('1970-71', t0."1970-71") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1975-76', t0."1975-76") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1980-81', t0."1980-81") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1985-86', t0."1985-86") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1990-91', t0."1990-91") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1995-96', t0."1995-96") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2000-01', t0."2000-01") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2005-06', t0."2005-06") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2010-11', t0."2010-11") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2011-12', t0."2011-12") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2012-13', t0."2012-13") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2013-14', t0."2013-14") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2014-15', t0."2014-15") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2015-16', t0."2015-16") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2016-17', t0."2016-17") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2017-18', t0."2017-18") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2018-19', t0."2018-19") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2019-20', t0."2019-20") AS Tuple(years Nullable(String), degrees Nullable(Int64)))]
          ).2 AS Nullable(Int64)) AS degrees
        FROM humanities AS t0
      ) AS t1
    ) AS t2
    GROUP BY
      t2.field_of_study
  ) AS t3
  WHERE
    t3.diff < 0
  ORDER BY
    t3.diff ASC
  LIMIT 10
) AS t8