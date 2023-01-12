SELECT t0.`string_col` AS `key`, CAST(t0.`float_col` AS double) AS `value`
FROM functional_alltypes t0
WHERE t0.`int_col` > 0
UNION
SELECT t0.`string_col` AS `key`, t0.`double_col` AS `value`
FROM functional_alltypes t0
WHERE t0.`int_col` <= 0