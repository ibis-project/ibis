SELECT
  indexOf(['a', 'b', 'c'], "t0"."string_col") - 1 AS "FindInSet(string_col, ('a', 'b', 'c'))"
FROM "functional_alltypes" AS "t0"