import ibis


first = ibis.table(
    name="first", schema={"key1": "string", "key2": "string", "value1": "float64"}
)
second = ibis.table(name="second", schema={"key1": "string", "value2": "float64"})
third = ibis.table(
    name="third", schema={"key2": "string", "key3": "string", "value3": "float64"}
)
fourth = ibis.table(name="fourth", schema={"key3": "string", "value4": "float64"})
joinchain = first.inner_join(second, first.key1 == second.key1)
joinchain1 = third.inner_join(fourth, third.key3 == fourth.key3)

result = joinchain.inner_join(joinchain1, joinchain.key2 == joinchain1.key2)
