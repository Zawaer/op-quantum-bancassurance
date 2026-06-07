# op_quantum_bancassurance_final.py

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ----------------------------
# CONFIG
# ----------------------------

N_CUSTOMERS = 20          # business demo size
N_IQM = 6                 # hardware-safe size; increase if backend allows
P_LAYERS = 1              # QAOA depth
SHOTS = 256
PENALTY_ONE_QUEUE = 8.0
PENALTY_CAPACITY = 3.0

RUN_IQM = False           # set True when IQM access is ready


# ----------------------------
# 1. Synthetic OP bancassurance data
# ----------------------------

def make_customers(n=20, seed=7):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "id": [f"C{i+1:02d}" for i in range(n)],
        "claim_estimate": rng.integers(5_000, 80_000, n),
        "set_aside_gap": rng.integers(0, 35_000, n),
        "op_bank_customer": rng.integers(0, 2, n),
        "mortgage_balance": rng.integers(0, 420_000, n),
        "payment_risk": rng.uniform(0.02, 0.90, n),
        "customer_value": rng.uniform(0.10, 1.00, n),
    })

    df.loc[df["op_bank_customer"] == 0, "mortgage_balance"] = 0
    return df


def normalize(x):
    m = max(float(np.max(x)), 1.0)
    return np.array(x) / m


customers = make_customers(N_CUSTOMERS)

customers["claim_n"] = normalize(customers["claim_estimate"])
customers["gap_n"] = normalize(customers["set_aside_gap"])
customers["mortgage_n"] = normalize(customers["mortgage_balance"])

customers["priority_score"] = (
    0.22 * customers["claim_n"]
    + 0.24 * customers["gap_n"]
    + 0.14 * customers["op_bank_customer"]
    + 0.15 * customers["mortgage_n"]
    + 0.17 * customers["payment_risk"]
    + 0.08 * customers["customer_value"]
)


# ----------------------------
# 2. Three queues
# y[i,0] = critical
# y[i,1] = fast
# y[i,2] = normal
# ----------------------------

QUEUE_NAMES = ["critical", "fast", "normal"]
QUEUE_SPEED_DAYS = np.array([1, 3, 7])
QUEUE_SERVICE_COST = np.array([900, 350, 80])

# More delay creates more financial damage.
def delay_cost(row, queue_idx):
    days = QUEUE_SPEED_DAYS[queue_idx]

    reserve_underestimation_cost = 0.18 * row["set_aside_gap"] * days
    claim_deterioration_cost = 0.025 * row["claim_estimate"] * days

    banking_cost = (
        row["op_bank_customer"]
        * row["payment_risk"]
        * (row["mortgage_balance"] * 0.00055)
        * days
    )

    churn_cost = row["customer_value"] * 1600 * days

    return (
        reserve_underestimation_cost
        + claim_deterioration_cost
        + banking_cost
        + churn_cost
        + QUEUE_SERVICE_COST[queue_idx]
    )


def total_business_cost(df, assignments):
    # assignments[i] in {0,1,2}
    total = 0.0
    for i, q in enumerate(assignments):
        total += delay_cost(df.iloc[i], q)
    return total


# ----------------------------
# 3. Classical actuarial baseline
# ----------------------------

def actuarial_baseline(df):
    # Traditional baseline:
    # rank mainly by claim estimate and set aside gap.
    baseline_score = (
        0.65 * normalize(df["claim_estimate"])
        + 0.35 * normalize(df["set_aside_gap"])
    )

    order = np.argsort(-baseline_score)

    assignments = np.full(len(df), 2)  # normal
    n_critical = max(1, len(df) // 5)
    n_fast = max(2, len(df) // 3)

    assignments[order[:n_critical]] = 0
    assignments[order[n_critical:n_critical + n_fast]] = 1

    return assignments


baseline_assignments = actuarial_baseline(customers)
baseline_cost = total_business_cost(customers, baseline_assignments)


# ----------------------------
# 4. QUBO construction
# variable index: v = 3*i + q
# ----------------------------

def build_qubo(df):
    n = len(df)
    m = 3 * n

    h = np.zeros(m)
    J = np.zeros((m, m))

    # Objective: choose queue with lowest total business cost.
    for i in range(n):
        for q in range(3):
            v = 3 * i + q
            h[v] += delay_cost(df.iloc[i], q)

    # Constraint: each customer must be assigned to exactly one queue.
    # penalty * (sum_q y_iq - 1)^2
    for i in range(n):
        vars_i = [3 * i + q for q in range(3)]

        for v in vars_i:
            h[v] += -PENALTY_ONE_QUEUE

        for a in range(3):
            for b in range(a + 1, 3):
                J[vars_i[a], vars_i[b]] += 2 * PENALTY_ONE_QUEUE

    # Capacity preference:
    # critical and fast queues should not take everyone.
    # This is soft, not hard.
    n_critical_target = max(1, n // 5)
    n_fast_target = max(2, n // 3)

    for q, target in [(0, n_critical_target), (1, n_fast_target)]:
        vars_q = [3 * i + q for i in range(n)]

        # penalty * (sum y - target)^2
        for v in vars_q:
            h[v] += PENALTY_CAPACITY * (1 - 2 * target)

        for a in range(len(vars_q)):
            for b in range(a + 1, len(vars_q)):
                J[vars_q[a], vars_q[b]] += 2 * PENALTY_CAPACITY

    return h, J


def bitstring_to_assignments(bitstring, n):
    bits = np.array([int(b) for b in bitstring[::-1]])
    assignments = []

    for i in range(n):
        group = bits[3*i:3*i+3]

        if group.sum() == 1:
            assignments.append(int(np.argmax(group)))
        else:
            # repair invalid quantum output:
            # choose the best queue by raw business cost
            costs = [delay_cost(active_df.iloc[i], q) for q in range(3)]
            assignments.append(int(np.argmin(costs)))

    return np.array(assignments)


def qubo_energy_from_bits(bits, h, J):
    x = np.array(bits)
    energy = float(np.dot(h, x))

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            energy += J[i, j] * x[i] * x[j]

    return energy


def qubo_energy_bitstring(bitstring, h, J):
    bits = [int(b) for b in bitstring[::-1]]
    return qubo_energy_from_bits(bits, h, J)


# ----------------------------
# 5. Warm start state
# ----------------------------

def assignment_to_bits(assignments):
    bits = np.zeros(3 * len(assignments), dtype=int)
    for i, q in enumerate(assignments):
        bits[3*i + q] = 1
    return bits


def build_warmstarted_qaoa(h, J, gammas, betas, warm_bits, warm_strength=0.85):
    n_qubits = len(h)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Warm start:
    # Instead of H on all qubits, rotate toward the classical baseline bitstring.
    # If warm bit = 1, prepare high probability of |1>.
    # If warm bit = 0, prepare high probability of |0>.
    theta_one = 2 * np.arcsin(np.sqrt(warm_strength))
    theta_zero = 2 * np.arcsin(np.sqrt(1 - warm_strength))

    for q in range(n_qubits):
        qc.ry(theta_one if warm_bits[q] == 1 else theta_zero, q)

    for gamma, beta in zip(gammas, betas):

        # Linear terms
        for i in range(n_qubits):
            if abs(h[i]) > 1e-12:
                qc.rz(-gamma * h[i], i)

        # Quadratic terms
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(gamma * J[i, j] / 2, i, j)

        # Mixer
        for q in range(n_qubits):
            qc.rx(2 * beta, q)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ----------------------------
# 6. QAOA solve
# ----------------------------

sim_backend = AerSimulator(method="matrix_product_state")
active_df = None


def solve_qaoa(df, initial_params=None, shots=SHOTS):
    global active_df
    active_df = df.reset_index(drop=True)

    h, J = build_qubo(active_df)

    warm_assignments = actuarial_baseline(active_df)
    warm_bits = assignment_to_bits(warm_assignments)

    if initial_params is None:
        initial_params = np.array([0.35] * P_LAYERS + [0.55] * P_LAYERS)

    def objective(params):
        gammas = params[:P_LAYERS]
        betas = params[P_LAYERS:]

        qc = build_warmstarted_qaoa(h, J, gammas, betas, warm_bits)
        result = sim_backend.run(qc, shots=shots).result()
        counts = result.get_counts()

        exp_energy = 0.0
        total = 0

        for bitstring, count in counts.items():
            exp_energy += qubo_energy_bitstring(bitstring, h, J) * count
            total += count

        return exp_energy / total

    result = minimize(
        objective,
        initial_params,
        method="COBYLA",
        options={"maxiter": 45}
    )

    final_params = result.x
    qc = build_warmstarted_qaoa(
        h,
        J,
        final_params[:P_LAYERS],
        final_params[P_LAYERS:],
        warm_bits
    )

    counts = sim_backend.run(qc, shots=4096).result().get_counts()

    best_bitstring = min(counts.keys(), key=lambda b: qubo_energy_bitstring(b, h, J))
    assignments = bitstring_to_assignments(best_bitstring, len(active_df))

    return {
        "params": final_params,
        "circuit": qc,
        "counts": counts,
        "bitstring": best_bitstring,
        "assignments": assignments,
        "cost": total_business_cost(active_df, assignments),
        "baseline_assignments": warm_assignments,
        "baseline_cost": total_business_cost(active_df, warm_assignments),
    }


# ----------------------------
# 7. Small QAOA + large optimized business demo
# ----------------------------

small_df = customers.head(N_IQM).copy()
small_solution = solve_qaoa(small_df)

baseline_assignments = actuarial_baseline(customers)

def optimized_business_assignment(df):
    """
    Large 20-customer business optimizer.
    Uses the same cost function as QUBO, but solves large instance classically
    so we can show financial impact without simulating 60 qubits.
    """
    n = len(df)

    n_critical = max(1, n // 5)
    n_fast = max(2, n // 3)

    # Cost if everyone is normal
    normal_costs = np.array([delay_cost(df.iloc[i], 2) for i in range(n)])
    fast_costs = np.array([delay_cost(df.iloc[i], 1) for i in range(n)])
    critical_costs = np.array([delay_cost(df.iloc[i], 0) for i in range(n)])

    # Benefit of moving from normal to fast / critical
    critical_saving = normal_costs - critical_costs
    fast_saving = normal_costs - fast_costs

    assignments = np.full(n, 2)

    # First assign biggest critical savings
    critical_order = np.argsort(-critical_saving)
    critical_idx = critical_order[:n_critical]
    assignments[critical_idx] = 0

    # Then assign fast among remaining customers
    remaining = [i for i in range(n) if i not in set(critical_idx)]
    remaining_sorted = sorted(remaining, key=lambda i: -fast_saving[i])
    fast_idx = remaining_sorted[:n_fast]
    assignments[fast_idx] = 1

    return assignments


optimized_assignments = optimized_business_assignment(customers)

baseline_cost = total_business_cost(customers, baseline_assignments)
optimized_cost = total_business_cost(customers, optimized_assignments)

saving = baseline_cost - optimized_cost
saving_pct = 100 * saving / baseline_cost

customers["baseline_queue"] = [QUEUE_NAMES[q] for q in baseline_assignments]
customers["optimized_queue"] = [QUEUE_NAMES[q] for q in optimized_assignments]

print("\n=== OP Quantum Bancassurance Demo ===")
print(f"Customers: {N_CUSTOMERS}")
print(f"Traditional actuarial baseline cost: €{baseline_cost:,.0f}")
print(f"QUBO-objective optimized cost:        €{optimized_cost:,.0f}")
print(f"Estimated saving:                    €{saving:,.0f}")
print(f"Estimated saving %:                  {saving_pct:.2f}%")

print("\nSmall QAOA/IQM-ready instance:")
print(f"Customers: {N_IQM}")
print(f"Small baseline cost: €{small_solution['baseline_cost']:,.0f}")
print(f"Small QAOA cost:     €{small_solution['cost']:,.0f}")

print("\nQueue comparison:")
print(customers[[
    "id",
    "claim_estimate",
    "set_aside_gap",
    "op_bank_customer",
    "mortgage_balance",
    "payment_risk",
    "baseline_queue",
    "optimized_queue"
]].sort_values(["optimized_queue", "claim_estimate"], ascending=[True, False]))






# ----------------------------
# 9. Optional IQM hardware run
# ----------------------------
# Before running:
# export IQM_SERVER_URL="https://..."
#
# Optional if your access requires auth:
# follow IQM event instructions for token/login.
#
# IQM docs use:
# from iqm.qiskit_iqm import IQMProvider
# provider = IQMProvider(IQM_SERVER_URL)
# backend = provider.get_backend()
# transpiled = transpile(circuit, backend)
# job = backend.run(transpiled, shots=...)
# result = job.result()

if RUN_IQM:
    from iqm.qiskit_iqm import IQMProvider

    IQM_SERVER_URL = os.environ["IQM_SERVER_URL"]

    provider = IQMProvider(IQM_SERVER_URL)
    iqm_backend = provider.get_backend()

    iqm_circuit = small_solution["circuit"]

    transpiled = transpile(iqm_circuit, iqm_backend)
    job = iqm_backend.run(transpiled, shots=SHOTS)
    iqm_result = job.result()
    iqm_counts = iqm_result.get_counts()

    print("\n=== IQM hardware counts ===")
    print(iqm_counts)

