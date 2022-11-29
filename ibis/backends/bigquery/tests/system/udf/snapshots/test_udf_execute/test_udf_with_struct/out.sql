CREATE TEMPORARY FUNCTION my_struct_thing_0(a FLOAT64, b FLOAT64)
RETURNS STRUCT<width FLOAT64, height FLOAT64>
LANGUAGE js AS """
'use strict';
function my_struct_thing(a, b) {
    class Rectangle {
        constructor(width, height) {
            this.width = width;
            this.height = height;
        }
    }
    return (new Rectangle(a, b));
}
return my_struct_thing(a, b);
""";