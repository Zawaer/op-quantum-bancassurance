# OP Quantum Bancassurance

**Quantum Hack 2026 · Espoo** — Team QuantumCollided

A quantum-classical hybrid optimizer for bancassurance claims processing, built for the OP Financial Group challenge.

## The problem

When a storm hits, an insurer faces hundreds of claims simultaneously. A standalone insurer prioritizes by claim size alone. OP is both a bank and an insurer — it sees mortgage balances and payment risk alongside claim data — but classical greedy algorithms can't jointly optimize across all that information at once.

## Our approach

We model the claims queue assignment as a **QUBO** (Quadratic Unconstrained Binary Optimization) problem with 60 binary variables (3 queues × 20 customers) and solve it using **QAOA** (Quantum Approximate Optimization Algorithm) warm-started from the actuarial baseline.

By treating banking and insurance data as a unified signal, the optimizer finds assignment combinations that classical greedy search misses — particularly customers whose mortgage exposure makes fast settlement critical even when the claim itself is modest.

**Result: 3.84% cost reduction** (€18,782 saved on a €489,344 baseline) on a 20-customer simulated storm scenario.

## Results

| | Baseline | Quantum-optimized |
|---|---|---|
| Total cost | €489,344 | €470,562 |
| Savings | — | **€18,782 (3.84%)** |
| Assignments changed | — | 6 / 20 customers |

## Stack

- **Qiskit** — QAOA circuit, COBYLA parameter optimization
- **AerSimulator** (`matrix_product_state`) — local simulation
- **Demo** — single-file HTML/JS interactive demo

## Files

```
algorithm.py      # QUBO formulation + QAOA optimizer (main implementation)
demo/index.html   # Interactive demo website
```

## Demo

Open `demo/index.html` in a browser, or visit the live deployment. Scroll through the scenario, then hit **Run QAOA Optimizer** to see the quantum assignments animate in.
