SELECT
  t3.key1 AS key1,
  AVG(t3.value1 - t3.value2) AS avg_diff
FROM (
  SELECT
    t0.value1 AS value1,
    t0.key1 AS key1,
    t0.key2 AS key2,
    t1.value2 AS value2,
    t1.key1 AS key1_right,
    t1.key4 AS key4
  FROM table1 AS t0
  LEFT OUTER JOIN table2 AS t1
    ON t0.key1 = t1.key1
) AS t3
GROUP BY
  1