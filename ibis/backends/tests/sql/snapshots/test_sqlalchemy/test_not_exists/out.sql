SELECT
  t0.key1,
  t0.key2,
  t0.value1
FROM foo_t AS t0
WHERE
  NOT (
    EXISTS(
      SELECT
        1 AS anon_1
      FROM bar_t AS t1
      WHERE
        t0.key1 = t1.key1
    )
  )