WITH "t1" AS (
  SELECT
    "t0"."street" AS "street",
    ROW_NUMBER() OVER (ORDER BY "t0"."street" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS "key"
  FROM "data" AS "t0"
), "t7" AS (
  SELECT
    "t6"."street" AS "street",
    ROW_NUMBER() OVER (ORDER BY "t6"."street" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS "key"
  FROM (
    SELECT
      "t3"."street" AS "street",
      "t3"."key" AS "key"
    FROM "t1" AS "t3"
    INNER JOIN (
      SELECT
        "t2"."key" AS "key"
      FROM "t1" AS "t2"
    ) AS "t5"
      ON "t3"."key" = "t5"."key"
  ) AS "t6"
)
SELECT
  "t9"."street" AS "street",
  "t9"."key" AS "key"
FROM "t7" AS "t9"
INNER JOIN (
  SELECT
    "t8"."key" AS "key"
  FROM "t7" AS "t8"
) AS "t11"
  ON "t9"."key" = "t11"."key"