SELECT
  CASE "t0"."string_col" WHEN 'foo' THEN 'bar' WHEN 'baz' THEN 'qux' ELSE 'default' END AS "SimpleCase(string_col, ('foo', 'baz'), ('bar', 'qux'), 'default')"
FROM "functional_alltypes" AS "t0"