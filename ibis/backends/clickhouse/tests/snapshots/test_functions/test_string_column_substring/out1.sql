SELECT
  SUBSTRING(
    "t0"."string_col",
    CASE WHEN (
      2 + 1
    ) >= 1 THEN 2 + 1 ELSE 2 + 1 + LENGTH("t0"."string_col") END
  ) AS "Substring(string_col, 2)"
FROM "functional_alltypes" AS "t0"