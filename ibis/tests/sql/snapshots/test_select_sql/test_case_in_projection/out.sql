SELECT
  CASE `g`
    WHEN 'foo' THEN 'bar'
    WHEN 'baz' THEN 'qux'
    ELSE 'default'
  END AS `col1`,
  CASE
    WHEN `g` = 'foo' THEN 'bar'
    WHEN `g` = 'baz' THEN `g`
    ELSE CAST(NULL AS string)
  END AS `col2`, *
FROM alltypes