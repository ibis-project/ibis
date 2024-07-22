WITH "t5" AS (
  SELECT
    "t4"."field_of_study",
    ANY_VALUE("t4"."diff") AS "diff"
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
            UNNEST(
              CAST([
                ROW('1970-71', "t0"."1970-71"),
                ROW('1975-76', "t0"."1975-76"),
                ROW('1980-81', "t0"."1980-81"),
                ROW('1985-86', "t0"."1985-86"),
                ROW('1990-91', "t0"."1990-91"),
                ROW('1995-96', "t0"."1995-96"),
                ROW('2000-01', "t0"."2000-01"),
                ROW('2005-06', "t0"."2005-06"),
                ROW('2010-11', "t0"."2010-11"),
                ROW('2011-12', "t0"."2011-12"),
                ROW('2012-13', "t0"."2012-13"),
                ROW('2013-14', "t0"."2013-14"),
                ROW('2014-15', "t0"."2014-15"),
                ROW('2015-16', "t0"."2015-16"),
                ROW('2016-17', "t0"."2016-17"),
                ROW('2017-18', "t0"."2017-18"),
                ROW('2018-19', "t0"."2018-19"),
                ROW('2019-20', "t0"."2019-20")
              ] AS STRUCT("years" TEXT, "degrees" BIGINT)[])
            ) AS "__pivoted__"
          FROM "humanities" AS "t0"
        ) AS "t1"
      ) AS "t2"
    ) AS "t3"
  ) AS "t4"
  GROUP BY
    1
)
SELECT
  *
FROM (
  SELECT
    *
  FROM "t5" AS "t6"
  ORDER BY
    "t6"."diff" DESC
  LIMIT 10
) AS "t9"
UNION ALL
SELECT
  *
FROM (
  SELECT
    *
  FROM "t5" AS "t6"
  WHERE
    "t6"."diff" < 0
  ORDER BY
    "t6"."diff" ASC
  LIMIT 10
) AS "t10"