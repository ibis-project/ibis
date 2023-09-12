function f() {
    let x = [[1, 2], [3, 4], [5, 6]].filter((([a, b]) => ((a > 1) && (b > 2)))).map((([a, b]) => (a + b)));
    return x;
}