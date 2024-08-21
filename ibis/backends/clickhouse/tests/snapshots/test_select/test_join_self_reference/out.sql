SELECT
  "t1"."id" AS "id",
  "t1"."bool_col" AS "bool_col",
  "t1"."tinyint_col" AS "tinyint_col",
  "t1"."smallint_col" AS "smallint_col",
  "t1"."int_col" AS "int_col",
  "t1"."bigint_col" AS "bigint_col",
  "t1"."float_col" AS "float_col",
  "t1"."double_col" AS "double_col",
  "t1"."date_string_col" AS "date_string_col",
  "t1"."string_col" AS "string_col",
  "t1"."timestamp_col" AS "timestamp_col",
  "t1"."year" AS "year",
  "t1"."month" AS "month"
FROM "functional_alltypes" AS "t1"
INNER JOIN "functional_alltypes" AS "t2"
  ON "t1"."id" = "t2"."id"