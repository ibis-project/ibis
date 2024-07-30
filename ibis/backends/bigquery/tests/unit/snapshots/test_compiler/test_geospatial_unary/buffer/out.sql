SELECT
  *
  REPLACE (st_asbinary(`tmp`) AS `tmp`)
FROM (
  SELECT
    st_buffer(`t0`.`geog`, 5.2) AS `tmp`
  FROM `t` AS `t0`
)