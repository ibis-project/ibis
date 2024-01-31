WITH "t5" AS (
  SELECT
    "t4"."field_of_study",
    FIRST("t4"."diff") AS "diff"
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
        FIRST("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "earliest_degrees",
        LAST("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "latest_degrees"
      FROM (
        SELECT
          "t1"."field_of_study",
          CAST(TO_JSONB("t1"."__pivoted__") -> 'f1' AS VARCHAR) AS "years",
          CAST(TO_JSONB("t1"."__pivoted__") -> 'f2' AS BIGINT) AS "degrees"
        FROM (
          SELECT
            "t0"."field_of_study",
            UNNEST(
              ARRAY[ROW(CAST('1970-71' AS VARCHAR), CAST("t0"."1970-71" AS BIGINT)), ROW(CAST('1975-76' AS VARCHAR), CAST("t0"."1975-76" AS BIGINT)), ROW(CAST('1980-81' AS VARCHAR), CAST("t0"."1980-81" AS BIGINT)), ROW(CAST('1985-86' AS VARCHAR), CAST("t0"."1985-86" AS BIGINT)), ROW(CAST('1990-91' AS VARCHAR), CAST("t0"."1990-91" AS BIGINT)), ROW(CAST('1995-96' AS VARCHAR), CAST("t0"."1995-96" AS BIGINT)), ROW(CAST('2000-01' AS VARCHAR), CAST("t0"."2000-01" AS BIGINT)), ROW(CAST('2005-06' AS VARCHAR), CAST("t0"."2005-06" AS BIGINT)), ROW(CAST('2010-11' AS VARCHAR), CAST("t0"."2010-11" AS BIGINT)), ROW(CAST('2011-12' AS VARCHAR), CAST("t0"."2011-12" AS BIGINT)), ROW(CAST('2012-13' AS VARCHAR), CAST("t0"."2012-13" AS BIGINT)), ROW(CAST('2013-14' AS VARCHAR), CAST("t0"."2013-14" AS BIGINT)), ROW(CAST('2014-15' AS VARCHAR), CAST("t0"."2014-15" AS BIGINT)), ROW(CAST('2015-16' AS VARCHAR), CAST("t0"."2015-16" AS BIGINT)), ROW(CAST('2016-17' AS VARCHAR), CAST("t0"."2016-17" AS BIGINT)), ROW(CAST('2017-18' AS VARCHAR), CAST("t0"."2017-18" AS BIGINT)), ROW(CAST('2018-19' AS VARCHAR), CAST("t0"."2018-19" AS BIGINT)), ROW(CAST('2019-20' AS VARCHAR), CAST("t0"."2019-20" AS BIGINT))]
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
  "t11"."field_of_study",
  "t11"."diff"
FROM (
  SELECT
    "t6"."field_of_study",
    "t6"."diff"
  FROM "t5" AS "t6"
  ORDER BY
    "t6"."diff" DESC NULLS LAST
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