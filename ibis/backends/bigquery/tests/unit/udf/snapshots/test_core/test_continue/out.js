function f() {
    let i = 0;
    for (let i of [1, 2, 3]) {
        if ((i === 1)) {
            continue;
        }
    }
    return i;
}