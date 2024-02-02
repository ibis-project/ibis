SELECT
  "t1"."id",
  "t1"."bool_col",
  "t1"."tinyint_col",
  "t1"."smallint_col",
  "t1"."int_col",
  "t1"."bigint_col",
  "t1"."float_col",
  "t1"."double_col",
  "t1"."date_string_col",
  "t1"."string_col",
  "t1"."timestamp_col",
  "t1"."year",
  "t1"."month"
FROM "functional_alltypes" AS "t1"
INNER JOIN "functional_alltypes" AS "t3"
  ON "t1"."id" = "t3"."id"