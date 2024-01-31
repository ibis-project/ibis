WITH "t1" AS (
  SELECT
    "t0"."region",
    "t0"."kind",
    SUM("t0"."amount") AS "total"
  FROM "purchases" AS "t0"
  GROUP BY
    1,
    2
)
SELECT
  "t5"."region",
  "t5"."total" - "t6"."total" AS "diff"
FROM (
  SELECT
    "t2"."region",
    "t2"."kind",
    "t2"."total"
  FROM "t1" AS "t2"
  WHERE
    "t2"."kind" = 'foo'
) AS "t5"
INNER JOIN (
  SELECT
    "t2"."region",
    "t2"."kind",
    "t2"."total"
  FROM "t1" AS "t2"
  WHERE
    "t2"."kind" = 'bar'
) AS "t6"
  ON "t5"."region" = "t6"."region"