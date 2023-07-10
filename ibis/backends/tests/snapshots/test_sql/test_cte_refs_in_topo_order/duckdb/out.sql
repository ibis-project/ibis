SELECT
  t1.key AS key,
  t2.key AS key_right,
  t5.key_right AS key_right_right
FROM (
  SELECT
    t0.key AS key
  FROM leaf AS t0
  WHERE
    TRUE
) AS t1
INNER JOIN (
  SELECT
    t0.key AS key
  FROM leaf AS t0
  WHERE
    TRUE
) AS t2
  ON t1.key = t2.key
INNER JOIN (
  SELECT
    t1.key AS key,
    t2.key AS key_right
  FROM (
    SELECT
      t0.key AS key
    FROM leaf AS t0
    WHERE
      TRUE
  ) AS t1
  INNER JOIN (
    SELECT
      t0.key AS key
    FROM leaf AS t0
    WHERE
      TRUE
  ) AS t2
    ON t1.key = t2.key
) AS t5
  ON t1.key = t5.key