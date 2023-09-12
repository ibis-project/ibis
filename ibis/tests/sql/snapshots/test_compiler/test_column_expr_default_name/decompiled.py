import ibis


int_col_table = ibis.table(name="int_col_table", schema={"int_col": "int32"})

result = int_col_table.int_col + 4
