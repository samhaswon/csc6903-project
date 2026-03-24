pragma circom 2.1.6;

/*
Minimal helper: decompose a number into bits.
Constrains in to be in [0, 2^n).
*/
template Num2Bits(n) {
    signal input in;
    signal output out[n];

    var lc = 0;
    var bitValue = 1;

    for (var i = 0; i < n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] - 1) === 0;

        lc += out[i] * bitValue;
        bitValue <<= 1;
    }

    lc === in;
}

/*
Returns 1 if a <= b, else 0.
Requires a,b in [0, 2^n).
*/
template LessEqThan(n) {
    signal input a;
    signal input b;
    signal output out;

    component bits = Num2Bits(n + 1);
    bits.in <== a + (1 << n) - b;
    out <== 1 - bits.out[n];
}

/*
True ZK proof of:
    |(last - first) - expected_change| <= tolerance

Representation:
- first_shifted = first + VALUE_SHIFT
- last_shifted = last + VALUE_SHIFT
- expected_shifted = expected_change + DELTA_SHIFT

Why two shifts:
- first and last are individual signed values
- delta = last - first is a signed value with a wider range

Choose parameters so that:
- first_shifted, last_shifted in [0, 2^BITS)
- delta_shifted, lower, upper in [0, 2^BITS)
- tolerance <= MAX_TOLERANCE
*/
template NetChange(BITS, VALUE_SHIFT, DELTA_SHIFT, MAX_TOLERANCE) {
    signal input first_shifted;
    signal input last_shifted;

    signal input expected_shifted;
    signal input tolerance;

    signal output ok;

    component firstBits = Num2Bits(BITS);
    component lastBits = Num2Bits(BITS);
    component expBits = Num2Bits(BITS);
    component tolBits = Num2Bits(BITS);

    firstBits.in <== first_shifted;
    lastBits.in <== last_shifted;
    expBits.in <== expected_shifted;
    tolBits.in <== tolerance;

    component tolLeMax = LessEqThan(BITS);
    tolLeMax.a <== tolerance;
    tolLeMax.b <== MAX_TOLERANCE;
    tolLeMax.out === 1;

    signal delta_shifted;
    signal lower;
    signal upper;

    delta_shifted <== last_shifted - first_shifted + DELTA_SHIFT;
    lower <== expected_shifted - tolerance;
    upper <== expected_shifted + tolerance;

    component deltaBits = Num2Bits(BITS);
    component lowerBits = Num2Bits(BITS);
    component upperBits = Num2Bits(BITS);

    deltaBits.in <== delta_shifted;
    lowerBits.in <== lower;
    upperBits.in <== upper;

    component lowerOk = LessEqThan(BITS);
    component upperOk = LessEqThan(BITS);

    lowerOk.a <== lower;
    lowerOk.b <== delta_shifted;

    upperOk.a <== delta_shifted;
    upperOk.b <== upper;

    ok <== lowerOk.out * upperOk.out;
    ok === 1;
}

/*
Example configuration.

Interpretation:
- private values are in about [-10^9, 10^9]
- expected net change is in about [-2*10^9, 2*10^9]
- tolerance is at most 10^9

These constants can be changed, but keep the inequalities valid.
*/
component main {public [expected_shifted, tolerance]} =
    NetChange(64, 1099511627776, 2199023255552, 1000000000);