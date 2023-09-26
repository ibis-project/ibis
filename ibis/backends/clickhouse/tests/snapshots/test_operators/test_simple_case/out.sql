SELECT
  CASE t0.string_col WHEN 'foo' THEN 'bar' WHEN 'baz' THEN 'qux' ELSE 'default' END AS "SimpleCase(string_col, 'default')"
FROM functional_alltypes AS t0