SELECT
  t4.key
FROM (
  SELECT
    t1.key
  FROM (
    SELECT
      *
    FROM leaf AS t0
    WHERE
      TRUE
  ) AS t1
  INNER JOIN (
    SELECT
      t1.key
    FROM (
      SELECT
        *
      FROM leaf AS t0
      WHERE
        TRUE
    ) AS t1
  ) AS t2
    ON t1.key = t2.key
) AS t4
INNER JOIN (
  SELECT
    t1.key
  FROM (
    SELECT
      *
    FROM leaf AS t0
    WHERE
      TRUE
  ) AS t1
  INNER JOIN (
    SELECT
      t1.key
    FROM (
      SELECT
        *
      FROM leaf AS t0
      WHERE
        TRUE
    ) AS t1
  ) AS t2
    ON t1.key = t2.key
) AS t5
  ON t4.key = t5.key