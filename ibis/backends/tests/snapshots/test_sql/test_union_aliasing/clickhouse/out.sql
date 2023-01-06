SELECT
  t11.field_of_study,
  t11.diff
FROM (
  SELECT
    *
  FROM (
    SELECT
      *
    FROM (
      SELECT
        t4.field_of_study,
        any(t4.diff) AS diff
      FROM (
        SELECT
          *,
          t3.latest_degrees - t3.earliest_degrees AS diff
        FROM (
          SELECT
            *,
            first_value(t2.degrees) OVER (PARTITION BY t2.field_of_study ORDER BY t2.years) AS earliest_degrees,
            last_value(t2.degrees) OVER (PARTITION BY t2.field_of_study ORDER BY t2.years) AS latest_degrees
          FROM (
            SELECT
              t1.field_of_study,
              CAST(t1.__pivoted__.1 AS Nullable(TEXT)) AS years,
              CAST(t1.__pivoted__.2 AS Nullable(Int64)) AS degrees
            FROM (
              SELECT
                t0.field_of_study,
                arrayJoin(
                  [  CAST(('1970-71', t0."1970-71") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('1975-76', t0."1975-76") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('1980-81', t0."1980-81") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('1985-86', t0."1985-86") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('1990-91', t0."1990-91") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('1995-96', t0."1995-96") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2000-01', t0."2000-01") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2005-06', t0."2005-06") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2010-11', t0."2010-11") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2011-12', t0."2011-12") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2012-13', t0."2012-13") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2013-14', t0."2013-14") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2014-15', t0."2014-15") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2015-16', t0."2015-16") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2016-17', t0."2016-17") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2017-18', t0."2017-18") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2018-19', t0."2018-19") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                    CAST(('2019-20', t0."2019-20") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64)))]
                ) AS __pivoted__
              FROM humanities AS t0
            ) AS t1
          ) AS t2
        ) AS t3
      ) AS t4
      GROUP BY
        t4.field_of_study
    ) AS t5
    ORDER BY
      t5.diff DESC
  ) AS t6
  LIMIT 10
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      *
    FROM (
      SELECT
        *
      FROM (
        SELECT
          t4.field_of_study,
          any(t4.diff) AS diff
        FROM (
          SELECT
            *,
            t3.latest_degrees - t3.earliest_degrees AS diff
          FROM (
            SELECT
              *,
              first_value(t2.degrees) OVER (PARTITION BY t2.field_of_study ORDER BY t2.years) AS earliest_degrees,
              last_value(t2.degrees) OVER (PARTITION BY t2.field_of_study ORDER BY t2.years) AS latest_degrees
            FROM (
              SELECT
                t1.field_of_study,
                CAST(t1.__pivoted__.1 AS Nullable(TEXT)) AS years,
                CAST(t1.__pivoted__.2 AS Nullable(Int64)) AS degrees
              FROM (
                SELECT
                  t0.field_of_study,
                  arrayJoin(
                    [  CAST(('1970-71', t0."1970-71") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('1975-76', t0."1975-76") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('1980-81', t0."1980-81") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('1985-86', t0."1985-86") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('1990-91', t0."1990-91") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('1995-96', t0."1995-96") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2000-01', t0."2000-01") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2005-06', t0."2005-06") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2010-11', t0."2010-11") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2011-12', t0."2011-12") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2012-13', t0."2012-13") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2013-14', t0."2013-14") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2014-15', t0."2014-15") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2015-16', t0."2015-16") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2016-17', t0."2016-17") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2017-18', t0."2017-18") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2018-19', t0."2018-19") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64))),
                      CAST(('2019-20', t0."2019-20") AS Tuple(years Nullable(TEXT), degrees Nullable(Int64)))]
                  ) AS __pivoted__
                FROM humanities AS t0
              ) AS t1
            ) AS t2
          ) AS t3
        ) AS t4
        GROUP BY
          t4.field_of_study
      ) AS t5
      WHERE
        t5.diff < 0
    ) AS t7
    ORDER BY
      t7.diff
  ) AS t9
  LIMIT 10
) AS t11