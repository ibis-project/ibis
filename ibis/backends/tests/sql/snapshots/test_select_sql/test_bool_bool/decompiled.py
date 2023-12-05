import ibis


airlines = ibis.table(
    name="airlines", schema={"dest": "string", "origin": "string", "arrdelay": "int32"}
)

result = airlines.filter((airlines.dest.cast("int64") == 0) == True)
