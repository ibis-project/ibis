SELECT
  "t2"."foo",
  "t2"."bar",
  "t2"."value",
  "t2"."baz",
  "t2"."foo" * CAST(2 AS TINYINT) AS "qux"
FROM (
  SELECT
    "t1"."foo",
    "t1"."bar",
    "t1"."value",
    "t1"."baz"
  FROM (
    SELECT
      "t0"."foo",
      "t0"."bar",
      "t0"."value",
      "t0"."foo" + "t0"."bar" AS "baz"
    FROM "tbl" AS "t0"
  ) AS "t1"
  WHERE
    "t1"."value" > CAST(0 AS TINYINT)
) AS "t2"