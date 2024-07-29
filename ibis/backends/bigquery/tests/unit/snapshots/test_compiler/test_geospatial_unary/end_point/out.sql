SELECT
  *
  REPLACE (st_asbinary(`tmp`) AS `tmp`)
FROM (
  SELECT
    st_endpoint(`t0`.`geog`) AS `tmp`
  FROM `t` AS `t0`
)