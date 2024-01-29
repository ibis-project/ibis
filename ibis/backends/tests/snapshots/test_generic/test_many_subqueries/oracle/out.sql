WITH "t6" AS (
  SELECT
    "t5"."street",
    ROW_NUMBER() OVER (ORDER BY "t5"."street" ASC NULLS LAST) - 1 AS "key"
  FROM (
    SELECT
      "t2"."street",
      "t2"."key"
    FROM (
      SELECT
        "t0"."street",
        ROW_NUMBER() OVER (ORDER BY "t0"."street" ASC NULLS LAST) - 1 AS "key"
      FROM "data" "t0"
    ) "t2"
    INNER JOIN (
      SELECT
        "t1"."key"
      FROM (
        SELECT
          "t0"."street",
          ROW_NUMBER() OVER (ORDER BY "t0"."street" ASC NULLS LAST) - 1 AS "key"
        FROM "data" "t0"
      ) "t1"
    ) "t4"
      ON "t2"."key" = "t4"."key"
  ) "t5"
), "t1" AS (
  SELECT
    "t0"."street",
    ROW_NUMBER() OVER (ORDER BY "t0"."street" ASC NULLS LAST) - 1 AS "key"
  FROM "data" "t0"
)
SELECT
  "t8"."street",
  "t8"."key"
FROM "t6" "t8"
INNER JOIN (
  SELECT
    "t7"."key"
  FROM "t6" "t7"
) "t10"
  ON "t8"."key" = "t10"."key"