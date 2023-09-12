function f(x, y, z) {
    function g(a, b, c) {
        return ((a - b) - c);
    }
    let args = [x, y, z];
    return g(...args);
}