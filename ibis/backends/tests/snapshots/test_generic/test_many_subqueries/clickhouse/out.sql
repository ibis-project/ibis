SELECT
  t5.street AS street,
  t5.key AS key,
  t5.key_right AS key_right
FROM (
  SELECT
    t1.street AS street,
    ROW_NUMBER() OVER (ORDER BY t1.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key,
    t3.key AS key_right
  FROM (
    SELECT
      t0.street AS street,
      ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key
    FROM data AS t0
  ) AS t1
  INNER JOIN (
    SELECT
      t1.key AS key
    FROM (
      SELECT
        t0.street AS street,
        ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key
      FROM data AS t0
    ) AS t1
  ) AS t3
    ON t1.key = t3.key
) AS t5
INNER JOIN (
  SELECT
    t5.key AS key
  FROM (
    SELECT
      t1.street AS street,
      ROW_NUMBER() OVER (ORDER BY t1.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key,
      t3.key AS key_right
    FROM (
      SELECT
        t0.street AS street,
        ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key
      FROM data AS t0
    ) AS t1
    INNER JOIN (
      SELECT
        t1.key AS key
      FROM (
        SELECT
          t0.street AS street,
          ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS key
        FROM data AS t0
      ) AS t1
    ) AS t3
      ON t1.key = t3.key
  ) AS t5
) AS t7
  ON t5.key = t7.key