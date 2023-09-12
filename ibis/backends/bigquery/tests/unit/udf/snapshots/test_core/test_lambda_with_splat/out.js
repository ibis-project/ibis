function f() {
    function sum(sequence) {
        let total = 0;
        for (let value of sequence) {
            total += value;
        }
        return total;
    }
    let splat_sum = ((...args) => sum(args));
    return splat_sum(1, 2, 3);
}