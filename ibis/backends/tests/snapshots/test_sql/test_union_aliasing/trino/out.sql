WITH "t5" AS (
  SELECT
    "t4"."field_of_study",
    ARBITRARY("t4"."diff") AS "diff"
  FROM (
    SELECT
      "t3"."field_of_study",
      "t3"."years",
      "t3"."degrees",
      "t3"."earliest_degrees",
      "t3"."latest_degrees",
      "t3"."latest_degrees" - "t3"."earliest_degrees" AS "diff"
    FROM (
      SELECT
        "t2"."field_of_study",
        "t2"."years",
        "t2"."degrees",
        FIRST_VALUE("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "earliest_degrees",
        LAST_VALUE("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "latest_degrees"
      FROM (
        SELECT
          "t1"."field_of_study",
          "t1"."__pivoted__"."years" AS "years",
          "t1"."__pivoted__"."degrees" AS "degrees"
        FROM (
          SELECT
            "t0"."field_of_study",
            IF(_u.pos = _u_2.pos_2, _u_2."__pivoted__") AS "__pivoted__"
          FROM "humanities" AS "t0"
          CROSS JOIN UNNEST(SEQUENCE(
            1,
            GREATEST(
              CARDINALITY(
                ARRAY[CAST(ROW('1970-71', "t0"."1970-71") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1975-76', "t0"."1975-76") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1980-81', "t0"."1980-81") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1985-86', "t0"."1985-86") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1990-91', "t0"."1990-91") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1995-96', "t0"."1995-96") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2000-01', "t0"."2000-01") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2005-06', "t0"."2005-06") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2010-11', "t0"."2010-11") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2011-12', "t0"."2011-12") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2012-13', "t0"."2012-13") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2013-14', "t0"."2013-14") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2014-15', "t0"."2014-15") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2015-16', "t0"."2015-16") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2016-17', "t0"."2016-17") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2017-18', "t0"."2017-18") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2018-19', "t0"."2018-19") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2019-20', "t0"."2019-20") AS ROW(years VARCHAR, degrees BIGINT))]
              )
            )
          )) AS _u(pos)
          CROSS JOIN UNNEST(ARRAY[CAST(ROW('1970-71', "t0"."1970-71") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1975-76', "t0"."1975-76") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1980-81', "t0"."1980-81") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1985-86', "t0"."1985-86") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1990-91', "t0"."1990-91") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1995-96', "t0"."1995-96") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2000-01', "t0"."2000-01") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2005-06', "t0"."2005-06") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2010-11', "t0"."2010-11") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2011-12', "t0"."2011-12") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2012-13', "t0"."2012-13") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2013-14', "t0"."2013-14") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2014-15', "t0"."2014-15") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2015-16', "t0"."2015-16") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2016-17', "t0"."2016-17") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2017-18', "t0"."2017-18") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2018-19', "t0"."2018-19") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2019-20', "t0"."2019-20") AS ROW(years VARCHAR, degrees BIGINT))]) WITH ORDINALITY AS _u_2("__pivoted__", pos_2)
          WHERE
            _u.pos = _u_2.pos_2
            OR (
              _u.pos > CARDINALITY(
                ARRAY[CAST(ROW('1970-71', "t0"."1970-71") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1975-76', "t0"."1975-76") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1980-81', "t0"."1980-81") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1985-86', "t0"."1985-86") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1990-91', "t0"."1990-91") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1995-96', "t0"."1995-96") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2000-01', "t0"."2000-01") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2005-06', "t0"."2005-06") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2010-11', "t0"."2010-11") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2011-12', "t0"."2011-12") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2012-13', "t0"."2012-13") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2013-14', "t0"."2013-14") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2014-15', "t0"."2014-15") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2015-16', "t0"."2015-16") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2016-17', "t0"."2016-17") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2017-18', "t0"."2017-18") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2018-19', "t0"."2018-19") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2019-20', "t0"."2019-20") AS ROW(years VARCHAR, degrees BIGINT))]
              )
              AND _u_2.pos_2 = CARDINALITY(
                ARRAY[CAST(ROW('1970-71', "t0"."1970-71") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1975-76', "t0"."1975-76") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1980-81', "t0"."1980-81") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1985-86', "t0"."1985-86") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1990-91', "t0"."1990-91") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('1995-96', "t0"."1995-96") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2000-01', "t0"."2000-01") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2005-06', "t0"."2005-06") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2010-11', "t0"."2010-11") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2011-12', "t0"."2011-12") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2012-13', "t0"."2012-13") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2013-14', "t0"."2013-14") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2014-15', "t0"."2014-15") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2015-16', "t0"."2015-16") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2016-17', "t0"."2016-17") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2017-18', "t0"."2017-18") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2018-19', "t0"."2018-19") AS ROW(years VARCHAR, degrees BIGINT)), CAST(ROW('2019-20', "t0"."2019-20") AS ROW(years VARCHAR, degrees BIGINT))]
              )
            )
        ) AS "t1"
      ) AS "t2"
    ) AS "t3"
  ) AS "t4"
  GROUP BY
    1
)
SELECT
  "t11"."field_of_study",
  "t11"."diff"
FROM (
  SELECT
    "t6"."field_of_study",
    "t6"."diff"
  FROM "t5" AS "t6"
  ORDER BY
    "t6"."diff" DESC
  LIMIT 10
  UNION ALL
  SELECT
    "t6"."field_of_study",
    "t6"."diff"
  FROM "t5" AS "t6"
  WHERE
    "t6"."diff" < 0
  ORDER BY
    "t6"."diff" ASC
  LIMIT 10
) AS "t11"