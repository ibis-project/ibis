SELECT
  t0.foo AS foo,
  t0.bar AS bar,
  t0.value AS value,
  t0.foo + t0.bar AS baz,
  t0.foo * CAST(2 AS TINYINT) AS qux
FROM tbl AS t0