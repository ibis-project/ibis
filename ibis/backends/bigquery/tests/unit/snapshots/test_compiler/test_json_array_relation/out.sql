WITH `items` AS (
  SELECT
    IF(pos = pos_2, `item`, NULL) AS `item`
  FROM `events` AS `t0`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(JSON_QUERY_ARRAY(PARSE_JSON(`t0`.`payload`)))) - 1)) AS pos
  CROSS JOIN UNNEST(JSON_QUERY_ARRAY(PARSE_JSON(`t0`.`payload`))) AS `item` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(JSON_QUERY_ARRAY(PARSE_JSON(`t0`.`payload`))) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(JSON_QUERY_ARRAY(PARSE_JSON(`t0`.`payload`))) - 1
      )
    )
)
SELECT
  safe.string(`t2`.`item`['name']) AS `name`,
  safe.string(`t2`.`item`['status']) AS `status`
FROM `items` AS `t2`