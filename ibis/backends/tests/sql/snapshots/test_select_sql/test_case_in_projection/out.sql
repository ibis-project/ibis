SELECT
  CASE "t0"."g" WHEN 'foo' THEN 'bar' WHEN 'baz' THEN 'qux' ELSE 'default' END AS "col1",
  CASE
    WHEN "t0"."g" = 'foo'
    THEN 'bar'
    WHEN "t0"."g" = 'baz'
    THEN "t0"."g"
    ELSE CAST(NULL AS TEXT)
  END AS "col2",
  "t0"."a",
  "t0"."b",
  "t0"."c",
  "t0"."d",
  "t0"."e",
  "t0"."f",
  "t0"."g",
  "t0"."h",
  "t0"."i",
  "t0"."j",
  "t0"."k"
FROM "alltypes" AS "t0"