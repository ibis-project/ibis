SELECT
  (
    t0.string_col LIKE 'foo%'
  ) OR (
    t0.string_col LIKE '%bar'
  ) AS "Or(StringSQLLike(string_col, 'foo%'), StringSQLLike(string_col, '%bar'))"
FROM functional_alltypes AS t0