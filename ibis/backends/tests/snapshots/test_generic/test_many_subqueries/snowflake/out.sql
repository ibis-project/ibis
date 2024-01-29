WITH "t6" AS (
  SELECT
    "t5"."street",
    ROW_NUMBER() OVER (ORDER BY "t5"."street" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS "key"
  FROM (
    SELECT
      "t2"."street",
      "t2"."key"
    FROM (
      SELECT
        "t0"."street",
        ROW_NUMBER() OVER (ORDER BY "t0"."street" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS "key"
      FROM "data" AS "t0"
    ) AS "t2"
    INNER JOIN (
      SELECT
        "t1"."key"
      FROM (
        SELECT
          "t0"."street",
          ROW_NUMBER() OVER (ORDER BY "t0"."street" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS "key"
        FROM "data" AS "t0"
      ) AS "t1"
    ) AS "t4"
      ON "t2"."key" = "t4"."key"
  ) AS "t5"
), "t1" AS (
  SELECT
    "t0"."street",
    ROW_NUMBER() OVER (ORDER BY "t0"."street" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS "key"
  FROM "data" AS "t0"
)
SELECT
  "t8"."street",
  "t8"."key"
FROM "t6" AS "t8"
INNER JOIN (
  SELECT
    "t7"."key"
  FROM "t6" AS "t7"
) AS "t10"
  ON "t8"."key" = "t10"."key"