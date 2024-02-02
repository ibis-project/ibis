SELECT
  "t4"."x1",
  "t4"."y1",
  "t5"."x2",
  "t9"."x3",
  "t9"."y2",
  "t9"."x4"
FROM "t1" AS "t4"
INNER JOIN "t2" AS "t5"
  ON "t4"."x1" = "t5"."x2"
INNER JOIN (
  SELECT
    "t6"."x3",
    "t6"."y2",
    "t7"."x4"
  FROM "t3" AS "t6"
  INNER JOIN "t4" AS "t7"
    ON "t6"."x3" = "t7"."x4"
) AS "t9"
  ON "t4"."y1" = "t9"."y2"