SELECT
  parse_timestamp('%F', `t0`.`date_string_col`, 'UTC') AS `StringToTimestamp_date_string_col_ '%F'`
FROM `functional_alltypes` AS `t0`