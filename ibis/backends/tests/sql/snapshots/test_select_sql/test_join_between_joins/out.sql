SELECT
  t6.key1 AS key1,
  t6.key2 AS key2,
  t6.value1 AS value1,
  t6.value2 AS value2,
  t7.value3 AS value3,
  t7.value4 AS value4
FROM (
  SELECT
    t0.key1 AS key1,
    t0.key2 AS key2,
    t0.value1 AS value1,
    t1.value2 AS value2
  FROM first AS t0
  INNER JOIN second AS t1
    ON t0.key1 = t1.key1
) AS t6
INNER JOIN (
  SELECT
    t2.key2 AS key2,
    t2.key3 AS key3,
    t2.value3 AS value3,
    t3.value4 AS value4
  FROM third AS t2
  INNER JOIN fourth AS t3
    ON t2.key3 = t3.key3
) AS t7
  ON t6.key2 = t7.key2