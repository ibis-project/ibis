SELECT
  *
  REPLACE (st_asbinary(`tmp`) AS `tmp`)
FROM (
  SELECT
    st_geogpoint(`t0`.`lon`, `t0`.`lat`) AS `tmp`
  FROM `t` AS `t0`
)