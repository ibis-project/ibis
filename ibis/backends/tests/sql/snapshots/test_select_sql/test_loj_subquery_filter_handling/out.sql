SELECT
  t2.id AS left_id,
  t2.desc AS left_desc,
  t3.id AS right_id,
  t3.desc AS right_desc
FROM (
  SELECT
    *
  FROM foo AS t0
  WHERE
    (
      t0.id < CAST(2 AS TINYINT)
    )
) AS t2
LEFT OUTER JOIN (
  SELECT
    *
  FROM bar AS t1
  WHERE
    (
      t1.id < CAST(3 AS TINYINT)
    )
) AS t3
  ON t2.id = t3.id AND t2.desc = t3.desc