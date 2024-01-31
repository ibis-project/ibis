SELECT
  COUNT(*) AS "CountStar()"
FROM (
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
    "t1"."month",
    "t3"."id" AS "id_right",
    "t3"."bool_col" AS "bool_col_right",
    "t3"."tinyint_col" AS "tinyint_col_right",
    "t3"."smallint_col" AS "smallint_col_right",
    "t3"."int_col" AS "int_col_right",
    "t3"."bigint_col" AS "bigint_col_right",
    "t3"."float_col" AS "float_col_right",
    "t3"."double_col" AS "double_col_right",
    "t3"."date_string_col" AS "date_string_col_right",
    "t3"."string_col" AS "string_col_right",
    "t3"."timestamp_col" AS "timestamp_col_right",
    "t3"."year" AS "year_right",
    "t3"."month" AS "month_right"
  FROM "functional_alltypes" AS "t1"
  INNER JOIN "functional_alltypes" AS "t3"
    ON "t1"."tinyint_col" < EXTRACT(minute FROM "t3"."timestamp_col")
) AS "t4"