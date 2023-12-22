SELECT
  t5.t1_id1,
  t5.t1_val1,
  t10.id3,
  t10.val2,
  t10.dt,
  t10.t3_val2,
  t10.id2a,
  t10.id2b,
  t10.val2_right
FROM (
  SELECT
    t0.id1 AS t1_id1,
    t0.val1 AS t1_val1
  FROM test1 AS t0
) AS t5
LEFT OUTER JOIN (
  SELECT
    t7.id3,
    t7.val2,
    t7.dt,
    t7.t3_val2,
    t3.id2a,
    t3.id2b,
    t3.val2 AS val2_right
  FROM (
    SELECT
      CAST(t1.id3 AS BIGINT) AS id3,
      t1.val2,
      t1.dt,
      CAST(t1.id3 AS BIGINT) AS t3_val2
    FROM test3 AS t1
  ) AS t7
  INNER JOIN test2 AS t3
    ON t3.id2b = t7.id3
) AS t10
  ON t5.t1_id1 = t10.id2a