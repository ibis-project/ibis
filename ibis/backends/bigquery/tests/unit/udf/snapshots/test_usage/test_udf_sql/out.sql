CREATE TEMPORARY FUNCTION format_t_0(
    input STRING
)
RETURNS FLOAT64 AS
(
  FORMAT('%T', input)
);

SELECT
  format_t_0('abcd') AS `format_t_0_'abcd'`