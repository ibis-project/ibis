SELECT
  t2.file_date,
  t2.PARTITIONTIME,
  t2.val,
  t2.XYZ
FROM (
  SELECT
    CAST(t0.file_date AS DATE) AS file_date,
    t0.PARTITIONTIME,
    t0.val,
    t0.val * 2 AS XYZ
  FROM unbound_table AS t0
  WHERE
    t0.PARTITIONTIME < DATE(2017, 1, 1) AND CAST(t0.file_date AS DATE) < DATE(2017, 1, 1)
) AS t2
INNER JOIN (
  SELECT
    CAST(t0.file_date AS DATE) AS file_date,
    t0.PARTITIONTIME,
    t0.val,
    t0.val * 2 AS XYZ
  FROM unbound_table AS t0
  WHERE
    t0.PARTITIONTIME < DATE(2017, 1, 1) AND CAST(t0.file_date AS DATE) < DATE(2017, 1, 1)
) AS t4
  ON TRUE