SELECT
  *
  REPLACE (st_asbinary(`tmp`) AS `tmp`)
FROM (
  SELECT
    st_pointn(`t0`.`geog`, 3) AS `tmp`
  FROM `t` AS `t0`
)