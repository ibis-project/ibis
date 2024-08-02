SELECT
  CASE
    WHEN empty(groupArray("t0"."string_col"))
    THEN NULL
    ELSE arrayStringConcat(groupArray("t0"."string_col"), '-')
  END AS "GroupConcat(string_col, '-', ())"
FROM "functional_alltypes" AS "t0"