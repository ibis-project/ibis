SELECT
  SUBSTRING(
    "t0"."string_col",
    CASE WHEN (
      0 + 1
    ) >= 1 THEN 0 + 1 ELSE 0 + 1 + LENGTH("t0"."string_col") END,
    3
  ) AS "Substring(string_col, 0, 3)"
FROM "functional_alltypes" AS "t0"