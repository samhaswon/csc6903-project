# Zero Knowledge Proof

This section of the repository details a zero-knowledge proof of demand reduction (or production) of the consumer side to the grid operator within a given tolerance. It does so without revealing the actual demand data of the consumer, which could compromise their privacy, particularly with the granularity of their data for this project.

The prover convinces the verifier that the statement is true without revealing the actual data values.

In total, the proof pipeline takes ~1s to prove and verify a statement.
```
Proof generation time: 0.51 seconds
Verification time: 0.41 seconds
```

It would likely be faster in something other than JS on the backend.

## Problem Statement

Let a private sequence of integers be:

$$
x_1, x_2, \ldots, x_n
$$

Define the net change:

$$
\Delta = x_n - x_1
$$

or

$$
\Delta = \frac{(\displaystyle\sum_{i=k}^{n} x_i}{k} - \frac{\displaystyle\sum_{i=0}^{k} x_i}{k}
$$

where $k$ is the point of the production/consumption-reduction request and $i = 0$ is the start of an appropriate number of historic samples.

The verifier supplies:

* an expected net change $e \in \mathbb{Z}$
* a tolerance $t \in \mathbb{Z}$ with $t \ge 0$

The goal is to prove:

$$
|\Delta - e| \le t
$$

without revealing $x_1$ or $x_n$ or their average equivalent.

## Algebraic Form

We avoid absolute values inside the circuit by rewriting:

$$
|\Delta - e| \le t
$$

as:

$$
-t \le (\Delta - e) \le t
$$

Substitute $\Delta = x_n - x_1$:

$$
-t \le (x_n - x_1 - e) \le t
$$

Rearranging:

$$
e - t \le (x_n - x_1) \le e + t
$$

This is the form enforced in the circuit.

## Handling Signed Integers

Circuits operate over a finite field, not over $\mathbb{Z}$. All values must be encoded as non-negative field elements.

We use shifted encodings.

### Endpoint encoding

Assume:

$$
x_1, x_n \in [-B, B]
$$

We encode:

$$
x_1' = x_1 + S
$$
$$
x_n' = x_n + S
$$

where $S$ is chosen such that:

$$
x_1', x_n' \in [0, 2^k)
$$

### Net change encoding

The net change lies in:

$$
\Delta \in [-2B, 2B]
$$

We encode:

$$
\Delta' = (x_n' - x_1') + S_\Delta
$$

where $S_\Delta$ ensures:

$$
\Delta' \ge 0
$$

### Expected value encoding

The verifier’s input is encoded as:

$$
e' = e + S_\Delta
$$

## Final Constraint in the Circuit

The circuit enforces:

$$
e' - t \le \Delta' \le e' + t
$$

This is equivalent to:

$$
|\Delta - e| \le t
$$

because both sides are shifted by the same constant.

## Circuit Structure

The circuit performs:

1. Range constraints
   Each value is decomposed into bits to ensure it lies within a valid range.
   This prevents wraparound in the finite field.

2. Compute shifted delta

$$
\Delta' = x_n' - x_1' + S_\Delta
$$

3. Compute bounds

$$
\text{lower} = e' - t
$$

$$
\text{upper} = e' + t
$$

4. Enforce inequalities

$$
\text{lower} \le \Delta'
$$

$$
\Delta' \le \text{upper}
$$

5. Constrain output

$$
\text{ok} = 1
$$

If any constraint fails, the proof cannot be generated.

## Zero-Knowledge Property

The proof reveals only:

* $e$ (expected net change)
* $t$ (tolerance)

It does not reveal:

* $x_1$
* $x_n$
* $\Delta$

This is achieved using a zk-SNARK (Groth16), which ensures:

* completeness
* soundness
* zero-knowledge

## Proof Workflow

### Setup (one-time)

* Compile the circuit to constraints
* Run trusted setup to produce proving and verification keys

### Prover

Given private inputs:

$$
x_1, x_n
$$

and public inputs:

$$
e, t
$$

the prover encodes the inputs, computes a witness, and generates a proof.

### Verifier

The verifier receives $(e, t)$ and the proof, verifies using the verification key, and accepts if valid.

## What This Proves

The proof establishes:

$$
\exists ~ x_1, x_n \text{ such that } |(x_n - x_1) - e| \le t
$$

It does not prove anything about:

* the intermediate values $x_2, \ldots, x_{n-1}$
* volatility or smoothness of before/after data
    * so the intermediate data could be a net change involving high out- or in-flows
* provenance of the data
    * the proof would happen on the meter, which we assume is trusted
    * consumers can face criminal charges for tampering with the meter
    * the security of smart meters is a hardware problem and/or otherwise outside of our scope for this project

## Limitations

* Only endpoint net change is proven
* Requires a trusted setup (Groth16)
* Range bounds must be chosen carefully
* Incorrect bounds can make the circuit unsound
