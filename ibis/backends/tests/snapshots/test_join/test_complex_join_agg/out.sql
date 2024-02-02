SELECT
  "t4"."key1",
  AVG("t4"."value1" - "t4"."value2") AS "avg_diff"
FROM (
  SELECT
    "t2"."value1",
    "t2"."key1",
    "t2"."key2",
    "t3"."value2",
    "t3"."key1" AS "key1_right",
    "t3"."key4"
  FROM "table1" AS "t2"
  LEFT OUTER JOIN "table2" AS "t3"
    ON "t2"."key1" = "t3"."key1"
) AS "t4"
GROUP BY
  1