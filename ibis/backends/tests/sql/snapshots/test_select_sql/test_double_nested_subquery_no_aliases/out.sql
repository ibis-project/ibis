SELECT
  "t2"."key1",
  SUM("t2"."total") AS "total"
FROM (
  SELECT
    "t1"."key1",
    "t1"."key2",
    SUM("t1"."total") AS "total"
  FROM (
    SELECT
      "t0"."key1",
      "t0"."key2",
      "t0"."key3",
      SUM("t0"."value") AS "total"
    FROM "foo_table" AS "t0"
    GROUP BY
      1,
      2,
      3
  ) AS "t1"
  GROUP BY
    1,
    2
) AS "t2"
GROUP BY
  1