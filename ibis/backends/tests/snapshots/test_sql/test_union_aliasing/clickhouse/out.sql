WITH "t5" AS (
  SELECT
    "t4"."field_of_study" AS "field_of_study",
    any("t4"."diff") AS "diff"
  FROM (
    SELECT
      "t3"."field_of_study" AS "field_of_study",
      "t3"."years" AS "years",
      "t3"."degrees" AS "degrees",
      "t3"."earliest_degrees" AS "earliest_degrees",
      "t3"."latest_degrees" AS "latest_degrees",
      "t3"."latest_degrees" - "t3"."earliest_degrees" AS "diff"
    FROM (
      SELECT
        "t2"."field_of_study" AS "field_of_study",
        "t2"."years" AS "years",
        "t2"."degrees" AS "degrees",
        FIRST_VALUE("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "earliest_degrees",
        LAST_VALUE("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "latest_degrees"
      FROM (
        SELECT
          "t1"."field_of_study" AS "field_of_study",
          CAST("t1"."__pivoted__".1 AS Nullable(String)) AS "years",
          CAST("t1"."__pivoted__".2 AS Nullable(Int64)) AS "degrees"
        FROM (
          SELECT
            "t0"."field_of_study" AS "field_of_study",
            arrayJoin(
              [
                tuple('1970-71', "t0"."1970-71"),
                tuple('1975-76', "t0"."1975-76"),
                tuple('1980-81', "t0"."1980-81"),
                tuple('1985-86', "t0"."1985-86"),
                tuple('1990-91', "t0"."1990-91"),
                tuple('1995-96', "t0"."1995-96"),
                tuple('2000-01', "t0"."2000-01"),
                tuple('2005-06', "t0"."2005-06"),
                tuple('2010-11', "t0"."2010-11"),
                tuple('2011-12', "t0"."2011-12"),
                tuple('2012-13', "t0"."2012-13"),
                tuple('2013-14', "t0"."2013-14"),
                tuple('2014-15', "t0"."2014-15"),
                tuple('2015-16', "t0"."2015-16"),
                tuple('2016-17', "t0"."2016-17"),
                tuple('2017-18', "t0"."2017-18"),
                tuple('2018-19', "t0"."2018-19"),
                tuple('2019-20', "t0"."2019-20")
              ]
            ) AS "__pivoted__"
          FROM "humanities" AS "t0"
        ) AS "t1"
      ) AS "t2"
    ) AS "t3"
  ) AS "t4"
  GROUP BY
    "t4"."field_of_study"
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