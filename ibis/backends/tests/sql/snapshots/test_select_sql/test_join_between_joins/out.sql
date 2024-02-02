SELECT
  "t4"."key1",
  "t4"."key2",
  "t4"."value1",
  "t5"."value2",
  "t9"."value3",
  "t9"."value4"
FROM "first" AS "t4"
INNER JOIN "second" AS "t5"
  ON "t4"."key1" = "t5"."key1"
INNER JOIN (
  SELECT
    "t6"."key2",
    "t6"."key3",
    "t6"."value3",
    "t7"."value4"
  FROM "third" AS "t6"
  INNER JOIN "fourth" AS "t7"
    ON "t6"."key3" = "t7"."key3"
) AS "t9"
  ON "t4"."key2" = "t9"."key2"