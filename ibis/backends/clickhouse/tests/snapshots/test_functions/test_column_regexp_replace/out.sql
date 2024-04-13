SELECT
  replaceRegexpAll("t0"."string_col", '[\\d]+', 'aaa') AS "RegexReplace(string_col, '[\\d]+', 'aaa')"
FROM "functional_alltypes" AS "t0"