WITH "t1" AS (
  SELECT
    "t0"."key"
  FROM "leaf" "t0"
  WHERE
    TRUE
)
SELECT
  "t3"."key"
FROM "t1" "t3"
INNER JOIN "t1" "t4"
  ON "t3"."key" = "t4"."key"
INNER JOIN (
  SELECT
    "t3"."key"
  FROM "t1" "t3"
  INNER JOIN "t1" "t4"
    ON "t3"."key" = "t4"."key"
) "t6"
  ON "t3"."key" = "t6"."key"