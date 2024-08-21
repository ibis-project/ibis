SELECT
  *
  REPLACE (st_asbinary(`tmp`) AS `tmp`)
FROM (
  SELECT
    st_startpoint(`t0`.`geog`) AS `tmp`
  FROM `t` AS `t0`
)