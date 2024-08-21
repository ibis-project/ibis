SELECT
  "t0"."foo",
  "t0"."bar",
  "t0"."value",
  "t0"."foo" + "t0"."bar" AS "baz",
  "t0"."foo" * 2 AS "qux"
FROM "tbl" AS "t0"
WHERE
  "t0"."value" > 0