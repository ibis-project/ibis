WITH "t1" AS (
  SELECT
    "t0"."street",
    ROW_NUMBER() OVER (ORDER BY "t0"."street" ASC) - 1 AS "key"
  FROM "data" "t0"
), "t7" AS (
  SELECT
    "t6"."street",
    ROW_NUMBER() OVER (ORDER BY "t6"."street" ASC) - 1 AS "key"
  FROM (
    SELECT
      "t3"."street",
      "t3"."key"
    FROM "t1" "t3"
    INNER JOIN (
      SELECT
        "t2"."key"
      FROM "t1" "t2"
    ) "t5"
      ON "t3"."key" = "t5"."key"
  ) "t6"
)
SELECT
  "t9"."street",
  "t9"."key"
FROM "t7" "t9"
INNER JOIN (
  SELECT
    "t8"."key"
  FROM "t7" "t8"
) "t11"
  ON "t9"."key" = "t11"."key"