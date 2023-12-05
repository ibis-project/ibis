import ibis


first = ibis.table(
    name="first", schema={"key1": "string", "key2": "string", "value1": "float64"}
)
third = ibis.table(
    name="third", schema={"key2": "string", "key3": "string", "value3": "float64"}
)
second = ibis.table(name="second", schema={"key1": "string", "value2": "float64"})
fourth = ibis.table(name="fourth", schema={"key3": "string", "value4": "float64"})
proj = first.inner_join(second, first.key1 == second.key1).select(
    [first, second.value2]
)
proj1 = third.inner_join(fourth, third.key3 == fourth.key3).select(
    [third, fourth.value4]
)

result = proj.inner_join(proj1, proj.key2 == proj1.key2).select(
    [proj, proj1.value3, proj1.value4]
)
