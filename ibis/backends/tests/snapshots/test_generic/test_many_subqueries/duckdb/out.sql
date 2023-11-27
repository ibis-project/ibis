SELECT
  t5.street AS street,
  t5.key AS key,
  t5.key_right AS key_right
FROM (
  SELECT
    t1.street AS street,
    ROW_NUMBER() OVER (ORDER BY t1.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - CAST(1 AS TINYINT) AS key,
    t2.key AS key_right
  FROM (
    SELECT
      t0.street AS street,
      ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - CAST(1 AS TINYINT) AS key
    FROM "data" AS t0
  ) AS t1
  INNER JOIN (
    SELECT
      t1.key AS key
    FROM (
      SELECT
        t0.street AS street,
        ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - CAST(1 AS TINYINT) AS key
      FROM "data" AS t0
    ) AS t1
  ) AS t2
    ON t1.key = t2.key
) AS t5
INNER JOIN (
  SELECT
    t5.key AS key
  FROM (
    SELECT
      t1.street AS street,
      ROW_NUMBER() OVER (ORDER BY t1.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - CAST(1 AS TINYINT) AS key,
      t2.key AS key_right
    FROM (
      SELECT
        t0.street AS street,
        ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - CAST(1 AS TINYINT) AS key
      FROM "data" AS t0
    ) AS t1
    INNER JOIN (
      SELECT
        t1.key AS key
      FROM (
        SELECT
          t0.street AS street,
          ROW_NUMBER() OVER (ORDER BY t0.street ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - CAST(1 AS TINYINT) AS key
        FROM "data" AS t0
      ) AS t1
    ) AS t2
      ON t1.key = t2.key
  ) AS t5
) AS t6
  ON t5.key = t6.key