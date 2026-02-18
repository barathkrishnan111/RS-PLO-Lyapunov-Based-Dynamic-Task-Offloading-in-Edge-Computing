"""
Lyapunov RS-PLO Engine — Risk-Sensitive Predictive Lyapunov Optimization
for real-time task offloading decisions in edge computing.

This module implements the full RS-PLO algorithm from the paper:
- Physical task queue Q(t)
- Virtual Volatility Queue Z(t) for channel risk tracking
- Adaptive control parameter V(t) = V_max * exp(-β * Z(t))
- Drift-Plus-Penalty per-slot offloading decisions
- Real task execution (local or offloaded to edge server)
"""

import socket
import time
import json
import struct
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
#  TCP CLIENT (for offloading to edge server)
# ─────────────────────────────────────────────

def send_message(sock: socket.socket, data: dict):
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_message(sock: socket.socket) -> dict:
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            return None
        raw_len += chunk
    msg_len = struct.unpack("!I", raw_len)[0]
    raw_data = b""
    while len(raw_data) < msg_len:
        chunk = sock.recv(msg_len - len(raw_data))
        if not chunk:
            return None
        raw_data += chunk
    return json.loads(raw_data.decode("utf-8"))


# ─────────────────────────────────────────────
#  LOCAL TASK EXECUTION
# ─────────────────────────────────────────────

def execute_locally(task_type: str, params: dict) -> dict:
    """Execute a task on the local device (IoT node / mobile device)."""
    start = time.perf_counter()

    if task_type == "matrix_multiply":
        size = params.get("size", 50)
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        C = np.dot(A, B)
        result = {"checksum": float(np.sum(C))}

    elif task_type == "hash_data":
        size_bytes = params.get("size_bytes", 50000)
        data = np.random.bytes(size_bytes)
        h = hashlib.sha256(data)
        for _ in range(100):
            h = hashlib.sha256(h.digest() + data[:1024])
        result = {"hash": h.hexdigest()[:16]}

    elif task_type == "prime_factorize":
        n = params.get("n", 999983)
        factors = []
        d = 2
        original = n
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        result = {"factors_count": len(factors), "largest": factors[-1] if factors else original}

    else:
        result = {"error": "unknown task"}

    elapsed = time.perf_counter() - start
    return {"exec_time": elapsed, "result": result, "location": "local"}


def execute_on_edge(task_type: str, params: dict, task_id: int,
                    host="127.0.0.1", port=9999) -> dict:
    """Offload a task to the edge server over TCP."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)

        connect_start = time.perf_counter()
        sock.connect((host, port))
        connect_time = time.perf_counter() - connect_start

        send_start = time.perf_counter()
        send_message(sock, {
            "task_type": task_type,
            "params": params,
            "task_id": task_id
        })

        response = recv_message(sock)
        total_time = time.perf_counter() - send_start
        sock.close()

        if response is None:
            return {"exec_time": total_time, "result": {}, "location": "edge", "error": "no response"}

        return {
            "exec_time": total_time,
            "server_exec_time": response.get("exec_time", 0),
            "network_overhead": total_time - response.get("exec_time", 0),
            "result": response,
            "location": "edge",
            "connect_time": connect_time
        }
    except Exception as e:
        return {"exec_time": 0, "result": {}, "location": "edge", "error": str(e)}


# ─────────────────────────────────────────────
#  SYSTEM PARAMETERS (from the paper)
# ─────────────────────────────────────────────

@dataclass
class SystemParams:
    """Configurable parameters for the RS-PLO framework."""
    # Number of users (IoT devices)
    N: int = 5

    # Maximum control parameter (energy weight)
    V_max: float = 10.0

    # Sensitivity parameter for V(t) decay
    beta: float = 2.0

    # Volatility queue decay rate (risk tolerance)
    gamma: float = 0.2

    # Task generation
    mean_task_size: int = 50       # mean matrix size (or data bytes)
    burst_probability: float = 0.2  # probability of a bursty slot
    burst_multiplier: float = 3.0   # task size multiplier during bursts

    # Energy model parameters
    # Local IoT device: tiny MCU, genuinely constrained
    f_local: float = 10e6           # Local CPU frequency (10 MHz — IoT microcontroller)
    kappa: float = 1e-28            # CPU capacitance coefficient
    local_energy_per_bit: float = 5e-6   # J/bit for local computation (slow CPU = more energy)

    # MEC edge server: powerful, but transmission costs energy
    P_tx: float = 0.5              # Transmission power (Watts)
    f_edge: float = 5e9            # Edge CPU frequency (5 GHz — powerful server)

    # Channel model
    h0: float = 1e-2               # Reference channel gain (at 1m)
    alpha: float = 3.0             # Path loss exponent (urban micro-cell)
    bandwidth: float = 10e6         # 10 MHz bandwidth
    noise_power: float = 1e-10     # Noise power spectral density

    # Simulation
    time_slots: int = 200           # Number of time slots
    edge_host: str = "127.0.0.1"
    edge_port: int = 9999

    # Task types to cycle through
    task_types: list = field(default_factory=lambda: ["matrix_multiply", "hash_data", "prime_factorize"])


# ─────────────────────────────────────────────
#  PER-USER STATE
# ─────────────────────────────────────────────

@dataclass
class UserState:
    """State for one IoT user/device."""
    user_id: int
    Q: float = 0.0               # Physical queue backlog (bits)
    Z: float = 0.0               # Virtual volatility queue
    prev_channel_gain: float = 0.0  # Previous channel gain (for prediction error)
    distance: float = 500.0       # Distance to BS in meters

    # Metrics history
    Q_history: list = field(default_factory=list)
    Z_history: list = field(default_factory=list)
    V_history: list = field(default_factory=list)
    energy_history: list = field(default_factory=list)
    decision_history: list = field(default_factory=list)  # 0=local, 1=offload
    latency_history: list = field(default_factory=list)
    channel_gain_history: list = field(default_factory=list)


# ─────────────────────────────────────────────
#  RS-PLO FRAMEWORK
# ─────────────────────────────────────────────

class RSPLO:
    """
    Risk-Sensitive Predictive Lyapunov Optimization engine.

    Makes real-time offloading decisions for each user at each time slot,
    actually executing tasks either locally or on the edge server.
    """

    def __init__(self, params: SystemParams, seed: int = 42):
        self.params = params
        self.users: List[UserState] = []
        self.global_task_id = 0
        self.slot = 0
        self.rng = np.random.RandomState(seed)

        # Initialize users with random distances
        for i in range(params.N):
            user = UserState(
                user_id=i,
                distance=self.rng.uniform(100, 600)  # 100m to 600m from BS
            )
            self.users.append(user)

    def compute_channel_gain(self, user: UserState) -> float:
        """
        Compute channel gain h(t) = h0 * d^(-alpha) * rayleigh_fading
        Real channel model with path loss + Rayleigh fading.
        """
        path_loss = self.params.h0 * (user.distance ** (-self.params.alpha))
        rayleigh = self.rng.exponential(1.0)  # Rayleigh fading
        return path_loss * rayleigh

    def compute_channel_gain_db(self, channel_gain: float) -> float:
        """Convert channel gain to dB scale for meaningful comparison."""
        if channel_gain <= 0:
            return -200.0
        return 10.0 * np.log10(channel_gain)

    def compute_transmission_rate(self, channel_gain: float) -> float:
        """Shannon capacity: R = B * log2(1 + P*h / N0)"""
        snr = self.params.P_tx * channel_gain / self.params.noise_power
        if snr <= 0:
            return 1.0
        return self.params.bandwidth * np.log2(1 + snr)

    def update_user_mobility(self, user: UserState):
        """
        Update user distance to BS — models real urban mobility.
        Random walk with occasional jumps and deep fade events.
        """
        delta = self.rng.normal(0, 15)  # normal movement (walk/slow drive)

        # Occasional jump (taxi turning corner, moving between streets)
        if self.rng.random() < 0.10:
            delta = self.rng.normal(0, 150)

        # Occasional deep fade event (entering parking garage / tunnel)
        if self.rng.random() < 0.04:
            delta = abs(self.rng.normal(500, 100))  # sudden move far away

        # Recovery: if very far, tendency to move back (user re-enters coverage)
        if user.distance > 1500:
            delta -= self.rng.uniform(50, 200)

        user.distance = np.clip(user.distance + delta, 30, 2500)

    def generate_task(self) -> dict:
        """Generate a real computation task with possible bursts."""
        is_burst = self.rng.random() < self.params.burst_probability

        task_type = self.rng.choice(self.params.task_types)

        if task_type == "matrix_multiply":
            base_size = self.params.mean_task_size
            size = int(base_size * (self.params.burst_multiplier if is_burst else 1.0))
            size = max(10, min(size, 200))
            params = {"size": size}
            task_bits = size * size * 64

        elif task_type == "hash_data":
            base_size = self.params.mean_task_size * 1000
            size_bytes = int(base_size * (self.params.burst_multiplier if is_burst else 1.0))
            params = {"size_bytes": size_bytes}
            task_bits = size_bytes * 8

        else:  # prime_factorize
            base_n = 999983
            n = base_n * (3 if is_burst else 1)
            params = {"n": n}
            task_bits = n.bit_length() * 1000

        return {
            "task_type": task_type,
            "params": params,
            "task_bits": task_bits,
            "is_burst": is_burst
        }

    def compute_V(self, Z: float) -> float:
        """
        Adaptive control parameter: V(t) = V_max * exp(-β * Z(t))

        When Z is low (stable) → V is high → prioritize energy saving
        When Z is high (volatile) → V drops → prioritize queue stability
        """
        return self.params.V_max * np.exp(-self.params.beta * Z)

    def compute_local_energy(self, task_bits: float) -> float:
        """
        Local energy consumption.
        E_local = local_energy_per_bit * task_bits
        Higher than offloading when channel is good (slow CPU, more cycles needed).
        """
        return self.params.local_energy_per_bit * task_bits

    def compute_offload_energy(self, task_bits: float, tx_rate: float) -> float:
        """
        Offload energy: E_offload = P_tx * (task_bits / R)
        Depends on channel quality (tx_rate). Good channel → low energy.
        Bad channel → very high energy (possibly higher than local).
        """
        if tx_rate <= 0:
            return float('inf')
        tx_time = task_bits / tx_rate
        return self.params.P_tx * tx_time

    def compute_local_service_rate(self) -> float:
        """
        Local processing service rate (bits/s).
        Limited IoT device CPU → service rate much lower than edge.
        """
        return self.params.f_local  # 500 MHz = 500M bits/s theoretical max

    def compute_edge_service_rate(self, channel_gain: float) -> float:
        """
        Edge service rate — bounded by the wireless transmission rate.
        Even though edge CPU is fast, the bottleneck is the wireless link.
        """
        return self.compute_transmission_rate(channel_gain)

    def drift_plus_penalty_decision(self, user: UserState, task: dict,
                                     channel_gain: float) -> int:
        """
        The core RS-PLO decision: minimize drift-plus-penalty.

        Uses normalized form so drift and penalty are on comparable scales:
          Cost(x) = drift_weight * drift_x + penalty_weight * V(t) * energy_x

        When V is HIGH (stable): energy penalty matters → pick lower energy
        When V is LOW (volatile): drift matters → pick higher service rate (μ)

        Returns: 0 = execute locally, 1 = offload to edge
        """
        V_t = self.compute_V(user.Z)
        Q_t = user.Q
        A_t = float(task["task_bits"])

        # ── Compute service rates ──
        mu_local = self.compute_local_service_rate()   # 10M bits/s (IoT MCU)
        mu_offload = self.compute_edge_service_rate(channel_gain)  # varies

        # ── Compute energy costs ──
        E_local = self.compute_local_energy(A_t)
        E_offload = self.compute_offload_energy(A_t, mu_offload)

        # ── Normalized Drift-Plus-Penalty ──
        # Drift: how much the queue benefits from this choice (normalized)
        # Higher service rate → lower (more negative) drift → better
        mu_max = max(mu_local, mu_offload, 1.0)
        drift_local = (A_t - mu_local) / mu_max      # normalized to [-1, ~0]
        drift_offload = (A_t - mu_offload) / mu_max

        # Penalty: energy cost (normalized by max energy)
        E_max = max(E_local, E_offload, 1e-10)
        penalty_local = E_local / E_max       # normalized to [0, 1]
        penalty_offload = E_offload / E_max

        # Queue-weighted drift + V-weighted energy penalty
        # Q_t amplifies the drift term (bigger queue → care more about service rate)
        # V(t) amplifies the penalty term (stable channel → care more about energy)
        Q_norm = Q_t / max(A_t, 1.0)  # queue size relative to task

        cost_local = Q_norm * drift_local + V_t * penalty_local
        cost_offload = Q_norm * drift_offload + V_t * penalty_offload

        if cost_offload < cost_local:
            return 1  # offload
        else:
            return 0  # local

    def run_slot(self, user: UserState) -> dict:
        """
        Execute one complete time slot for a user.
        """
        self.global_task_id += 1

        # 1. Update mobility
        self.update_user_mobility(user)

        # 2. Generate task
        task = self.generate_task()

        # 3. Compute channel gain
        channel_gain = self.compute_channel_gain(user)
        channel_gain_db = self.compute_channel_gain_db(channel_gain)

        # 4. Compute NORMALIZED prediction error and update Volatility Queue Z(t)
        if user.prev_channel_gain > 0:
            prev_db = self.compute_channel_gain_db(user.prev_channel_gain)
            # Normalized prediction error in dB
            prediction_error = abs(channel_gain_db - prev_db) / 10.0
        else:
            prediction_error = 0.0
        user.prev_channel_gain = channel_gain

        # Z(t+1) = max(Z(t) - γ, 0) + e(t)
        user.Z = max(user.Z - self.params.gamma, 0) + prediction_error

        # 5. Compute adaptive V(t)
        V_t = self.compute_V(user.Z)

        # 6. Make offloading decision
        decision = self.drift_plus_penalty_decision(user, task, channel_gain)

        # 7. ACTUALLY EXECUTE THE TASK
        if decision == 1:
            result = execute_on_edge(
                task["task_type"], task["params"],
                self.global_task_id,
                self.params.edge_host, self.params.edge_port
            )
            if "error" in result and result.get("error"):
                result = execute_locally(task["task_type"], task["params"])
                decision = 0
        else:
            result = execute_locally(task["task_type"], task["params"])

        exec_time = result["exec_time"]

        # 8. Compute actual energy consumed
        tx_rate = self.compute_transmission_rate(channel_gain)
        if decision == 1:
            energy = self.compute_offload_energy(task["task_bits"], tx_rate)
        else:
            energy = self.compute_local_energy(task["task_bits"])

        # 9. Update physical queue: Q(t+1) = max(Q(t) - μ(t) * Δt, 0) + A(t)
        if decision == 1:
            service_bits = tx_rate * exec_time
        else:
            service_bits = self.params.f_local * exec_time

        user.Q = max(user.Q - service_bits, 0) + task["task_bits"]

        # 10. Record metrics
        user.Q_history.append(user.Q)
        user.Z_history.append(user.Z)
        user.V_history.append(V_t)
        user.energy_history.append(energy)
        user.decision_history.append(decision)
        user.latency_history.append(exec_time)
        user.channel_gain_history.append(channel_gain_db)

        return {
            "slot": self.slot,
            "user_id": user.user_id,
            "task_type": task["task_type"],
            "is_burst": task["is_burst"],
            "decision": "OFFLOAD" if decision == 1 else "LOCAL",
            "exec_time_ms": exec_time * 1000,
            "energy_mJ": energy * 1000,
            "Q": user.Q,
            "Z": user.Z,
            "V": V_t,
            "channel_gain_db": channel_gain_db,
            "distance_m": user.distance
        }

    def run_all_slots(self, verbose=True) -> dict:
        """Run the full experiment for all users across all time slots."""
        all_results = []

        for t in range(self.params.time_slots):
            self.slot = t
            slot_results = []

            for user in self.users:
                result = self.run_slot(user)
                slot_results.append(result)

                if verbose and t % 20 == 0 and user.user_id == 0:
                    print(f"  [Slot {t:3d}] User 0: {result['decision']:8s} | "
                          f"Q={result['Q']:>10.0f} | Z={result['Z']:.3f} | "
                          f"V={result['V']:.3f} | h={result['channel_gain_db']:.1f}dB | "
                          f"d={result['distance_m']:.0f}m | E={result['energy_mJ']:.3f}mJ | "
                          f"{result['task_type']} "
                          f"{'⚡BURST' if result['is_burst'] else ''}")

            all_results.append(slot_results)

        return self._aggregate_results()

    def _aggregate_results(self) -> dict:
        """Aggregate per-user metrics into summary statistics."""
        summary = {
            "users": [],
            "avg_Q": 0, "avg_Z": 0, "avg_V": 0,
            "total_energy": 0, "offload_ratio": 0,
            "avg_latency_ms": 0
        }

        total_decisions = 0
        total_offloads = 0

        for user in self.users:
            user_summary = {
                "user_id": user.user_id,
                "avg_Q": np.mean(user.Q_history) if user.Q_history else 0,
                "max_Q": np.max(user.Q_history) if user.Q_history else 0,
                "avg_Z": np.mean(user.Z_history) if user.Z_history else 0,
                "max_Z": np.max(user.Z_history) if user.Z_history else 0,
                "avg_V": np.mean(user.V_history) if user.V_history else 0,
                "min_V": np.min(user.V_history) if user.V_history else 0,
                "total_energy": np.sum(user.energy_history) if user.energy_history else 0,
                "offload_ratio": np.mean(user.decision_history) if user.decision_history else 0,
                "avg_latency_ms": np.mean(user.latency_history) * 1000 if user.latency_history else 0,
                "Q_history": user.Q_history,
                "Z_history": user.Z_history,
                "V_history": user.V_history,
                "energy_history": user.energy_history,
                "decision_history": user.decision_history,
                "latency_history": user.latency_history,
                "channel_gain_history": user.channel_gain_history,
            }
            summary["users"].append(user_summary)
            total_decisions += len(user.decision_history)
            total_offloads += sum(user.decision_history)

        summary["avg_Q"] = np.mean([u["avg_Q"] for u in summary["users"]])
        summary["avg_Z"] = np.mean([u["avg_Z"] for u in summary["users"]])
        summary["avg_V"] = np.mean([u["avg_V"] for u in summary["users"]])
        summary["total_energy"] = np.sum([u["total_energy"] for u in summary["users"]])
        summary["offload_ratio"] = total_offloads / total_decisions if total_decisions > 0 else 0
        summary["avg_latency_ms"] = np.mean([u["avg_latency_ms"] for u in summary["users"]])

        return summary


# ─────────────────────────────────────────────
#  STATIC LYAPUNOV BASELINE (for comparison)
# ─────────────────────────────────────────────

class StaticLyapunov(RSPLO):
    """
    Baseline: Standard Lyapunov with FIXED V (no adaptivity).
    Same as RS-PLO but V(t) = V_max always, ignoring volatility.
    """

    def compute_V(self, Z: float) -> float:
        """Override: V is always static — ignores channel volatility."""
        return self.params.V_max
