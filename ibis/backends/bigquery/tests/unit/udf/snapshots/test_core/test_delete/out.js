function f(a) {
    let x = [a, 1, 2, 3];
    let y = {['a']: 1};
    delete x[0];
    delete x[1];
    delete x[(0 + 3)];
    delete y.a;
    return 1;
}