SELECT
  CASE t0.`g`
    WHEN 'foo' THEN 'bar'
    WHEN 'baz' THEN 'qux'
    ELSE 'default'
  END AS `col1`,
  CASE
    WHEN t0.`g` = 'foo' THEN 'bar'
    WHEN t0.`g` = 'baz' THEN t0.`g`
    ELSE CAST(NULL AS string)
  END AS `col2`, t0.*
FROM alltypes t0