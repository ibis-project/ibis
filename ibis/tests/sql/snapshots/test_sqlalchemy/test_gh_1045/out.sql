SELECT
  t0.t1_id1,
  t0.t1_val1,
  t1.dt,
  t1.id3,
  t1.t3_val2,
  t1.id2a,
  t1.id2b,
  t1.val2
FROM (
  SELECT
    t2.id1 AS t1_id1,
    t2.val1 AS t1_val1
  FROM test1 AS t2
) AS t0
LEFT OUTER JOIN (
  SELECT
    t2.dt AS dt,
    t2.id3 AS id3,
    t2.t3_val2 AS t3_val2,
    t3.id2a AS id2a,
    t3.id2b AS id2b,
    t3.val2 AS val2
  FROM (
    SELECT
      t4.dt AS dt,
      t4.id3 AS id3,
      t4.id3 AS t3_val2
    FROM (
      SELECT
        t5.val2 AS val2,
        t5.dt AS dt,
        CAST(t5.id3 AS BIGINT) AS id3
      FROM test3 AS t5
    ) AS t4
  ) AS t2
  JOIN test2 AS t3
    ON t3.id2b = t2.id3
) AS t1
  ON t0.t1_id1 = t1.id2a