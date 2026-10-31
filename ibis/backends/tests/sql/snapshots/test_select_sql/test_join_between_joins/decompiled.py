import ibis

first = ibis.table(
    name="first", schema={"key1": "string", "key2": "string", "value1": "float64"}
)
second = ibis.table(name="second", schema={"key1": "string", "value2": "float64"})
third = ibis.table(
    name="third", schema={"key2": "string", "key3": "string", "value3": "float64"}
)
fourth = ibis.table(name="fourth", schema={"key3": "string", "value4": "float64"})
joinchain = third.inner_join(fourth, (third.key3 == fourth.key3)).select(
    third.key2, third.key3, third.value3, fourth.value4
)

result = (
    first.inner_join(second, (first.key1 == second.key1))
    .inner_join(joinchain, (first.key2 == joinchain.key2))
    .select(
        first.key1,
        first.key2,
        first.value1,
        second.value2,
        joinchain.value3,
        joinchain.value4,
    )
)
