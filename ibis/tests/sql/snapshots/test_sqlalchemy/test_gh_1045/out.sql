WITH t2 AS (
  SELECT
    t4.id1 AS t1_id1,
    t4.val1 AS t1_val1
  FROM test1 AS t4
), t0 AS (
  SELECT
    t4.val2 AS val2,
    t4.dt AS dt,
    CAST(t4.id3 AS BIGINT) AS id3
  FROM test3 AS t4
), t1 AS (
  SELECT
    t0.dt AS dt,
    t0.id3 AS id3,
    t0.id3 AS t3_val2
  FROM t0
)
SELECT
  t2.t1_id1,
  t2.t1_val1,
  t3.dt,
  t3.id3,
  t3.t3_val2,
  t3.id2a,
  t3.id2b,
  t3.val2
FROM t2
LEFT OUTER JOIN (
  SELECT
    t1.dt AS dt,
    t1.id3 AS id3,
    t1.t3_val2 AS t3_val2,
    t4.id2a AS id2a,
    t4.id2b AS id2b,
    t4.val2 AS val2
  FROM t1
  JOIN test2 AS t4
    ON t4.id2b = t1.id3
) AS t3
  ON t2.t1_id1 = t3.id2a