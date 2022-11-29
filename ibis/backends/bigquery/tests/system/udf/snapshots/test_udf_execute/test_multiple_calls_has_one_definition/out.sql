CREATE TEMPORARY FUNCTION my_str_len_0(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_str_len(s) {
    return s.length;
}
return my_str_len(s);
""";

SELECT my_str_len_0('abcd') + my_str_len_0('abcd') AS `tmp`