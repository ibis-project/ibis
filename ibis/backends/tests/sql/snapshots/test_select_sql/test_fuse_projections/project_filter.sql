SELECT
  t1.foo AS foo,
  t1.bar AS bar,
  t1.value AS value,
  t1.foo + t1.bar AS baz,
  t1.foo * CAST(2 AS TINYINT) AS qux
FROM (
  SELECT
    *
  FROM tbl AS t0
  WHERE
    (
      t0.value > CAST(0 AS TINYINT)
    )
) AS t1