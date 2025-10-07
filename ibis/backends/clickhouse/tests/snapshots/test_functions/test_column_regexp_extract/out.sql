SELECT
  CASE
    WHEN notEmpty(extractGroups(CAST("t0"."string_col" AS String), CONCAT('(', '[\\d]+', ')'))[4])
    THEN extractGroups(CAST("t0"."string_col" AS String), CONCAT('(', '[\\d]+', ')'))[4]
    ELSE NULL
  END AS "RegexExtract(string_col, '[\\d]+', 3)"
FROM "functional_alltypes" AS "t0"