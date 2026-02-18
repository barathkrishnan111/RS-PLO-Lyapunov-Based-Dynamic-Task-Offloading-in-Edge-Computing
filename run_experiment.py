"""
Run Experiment — Orchestrates the full RS-PLO task offloading experiment.

1. Starts the edge server as a subprocess
2. Runs RS-PLO algorithm (adaptive V) for T time slots
3. Runs Static Lyapunov baseline (fixed V) for comparison
4. Produces Matplotlib plots comparing both approaches
5. Prints summary statistics

Usage:  python run_experiment.py
"""

import subprocess
import sys
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from lyapunov_engine import RSPLO, StaticLyapunov, SystemParams


def start_edge_servers(ports=[9999, 10000, 10001]) -> list:
    """Start multiple edge servers as background subprocesses."""
    print("=" * 70)
    print("  LYAPUNOV-BASED TASK OFFLOADING IN EDGE COMPUTING")
    print("  RS-PLO: Risk-Sensitive Predictive Lyapunov Optimization")
    print("  Multi-Edge Server Mode (3 servers)")
    print("=" * 70)
    print()
    print(f"[1/5] Starting {len(ports)} Edge Servers...")

    procs = []
    for port in ports:
        proc = subprocess.Popen(
            [sys.executable, "edge_server.py", str(port)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        procs.append(proc)

    time.sleep(2)  # Wait for servers to start

    for i, proc in enumerate(procs):
        if proc.poll() is not None:
            print(f"[ERROR] Edge server on port {ports[i]} failed to start!")
            sys.exit(1)
        print(f"       Edge-{i+1} running on 127.0.0.1:{ports[i]} (PID: {proc.pid})")

    return procs


def run_rsplo(params: SystemParams) -> dict:
    """Run the RS-PLO (adaptive) algorithm."""
    print()
    print("[2/5] Running RS-PLO (Adaptive Lyapunov)...")
    print(f"       Users: {params.N} | Time Slots: {params.time_slots} | "
          f"V_max: {params.V_max} | β: {params.beta} | Burst Prob: {params.burst_probability}")
    print()

    engine = RSPLO(params)
    results = engine.run_all_slots(verbose=True)

    print()
    print(f"       ✓ RS-PLO complete. Avg Queue: {results['avg_Q']:.1f} | "
          f"Offload Ratio: {results['offload_ratio']:.1%} | "
          f"Avg Latency: {results['avg_latency_ms']:.2f}ms")

    return results


def run_static_baseline(params: SystemParams) -> dict:
    """Run the Static Lyapunov baseline (fixed V)."""
    print()
    print("[3/5] Running Static Lyapunov Baseline (fixed V)...")

    engine = StaticLyapunov(params)
    results = engine.run_all_slots(verbose=False)

    print(f"       ✓ Static baseline complete. Avg Queue: {results['avg_Q']:.1f} | "
          f"Offload Ratio: {results['offload_ratio']:.1%} | "
          f"Avg Latency: {results['avg_latency_ms']:.2f}ms")

    return results


def generate_plots(rsplo_results: dict, static_results: dict, output_dir: str):
    """Generate comparison plots — the key evidence of RS-PLO working."""
    print()
    print("[4/5] Generating comparison plots...")

    os.makedirs(output_dir, exist_ok=True)

    # Use User 0 for detailed plots
    rsplo_u0 = rsplo_results["users"][0]
    static_u0 = static_results["users"][0]
    slots = range(len(rsplo_u0["Q_history"]))

    # Color scheme
    CYAN = "#00D4FF"
    RED = "#FF4444"
    GREEN = "#00FF88"
    PURPLE = "#B266FF"
    YELLOW = "#FFD700"
    BG = "#0A0E17"
    PANEL = "#151B2B"
    TEXT = "#E0E0E0"

    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': PANEL,
        'axes.edgecolor': '#2A3050',
        'axes.labelcolor': TEXT,
        'text.color': TEXT,
        'xtick.color': TEXT,
        'ytick.color': TEXT,
        'grid.color': '#1E2640',
        'grid.alpha': 0.5,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })

    # ── FIGURE 1: RS-PLO Internal Mechanism (3 subplots) ──
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig1.suptitle("RS-PLO Internal Mechanism — Real-Time Adaptation", fontsize=16, fontweight='bold', y=0.98)

    # Physical Queue Q(t)
    ax1.plot(slots, rsplo_u0["Q_history"], color=CYAN, linewidth=1.5, label="Q(t) — Physical Queue")
    ax1.fill_between(slots, rsplo_u0["Q_history"], alpha=0.15, color=CYAN)
    ax1.set_ylabel("Queue Backlog (bits)")
    ax1.set_title("Physical Task Queue Q(t)")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Volatility Queue Z(t)
    ax2.plot(slots, rsplo_u0["Z_history"], color=RED, linewidth=1.5, label="Z(t) — Volatility Queue")
    ax2.fill_between(slots, rsplo_u0["Z_history"], alpha=0.15, color=RED)
    ax2.set_ylabel("Volatility Level")
    ax2.set_title("Virtual Volatility Queue Z(t) — Risk Accumulator")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Adaptive V(t)
    ax3.plot(slots, rsplo_u0["V_history"], color=GREEN, linewidth=1.5, label="V(t) — Adaptive Control")
    ax3.fill_between(slots, rsplo_u0["V_history"], alpha=0.15, color=GREEN)
    ax3.set_xlabel("Time Slot")
    ax3.set_ylabel("V(t)")
    ax3.set_title("Adaptive Control Parameter V(t) = V_max · exp(−β · Z(t))")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle='--', alpha=0.3)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(output_dir, "01_rsplo_mechanism.png"), dpi=150, bbox_inches='tight')
    print(f"       → Saved: {output_dir}/01_rsplo_mechanism.png")

    # ── FIGURE 2: RS-PLO vs Static Comparison ──
    fig2, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle("RS-PLO vs Static Lyapunov — Performance Comparison", fontsize=16, fontweight='bold', y=0.98)

    # Queue Comparison
    ax4.plot(slots, rsplo_u0["Q_history"], color=CYAN, linewidth=1.5, label="RS-PLO (Adaptive)")
    ax4.plot(slots, static_u0["Q_history"], color=RED, linewidth=1.5, linestyle='--', label="Static Lyapunov", alpha=0.8)
    ax4.set_xlabel("Time Slot")
    ax4.set_ylabel("Queue Backlog (bits)")
    ax4.set_title("Physical Queue Comparison")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.3)

    # Energy Comparison (cumulative)
    rsplo_cum_energy = np.cumsum(rsplo_u0["energy_history"])
    static_cum_energy = np.cumsum(static_u0["energy_history"])
    ax5.plot(slots, rsplo_cum_energy, color=GREEN, linewidth=1.5, label="RS-PLO")
    ax5.plot(slots, static_cum_energy, color=RED, linewidth=1.5, linestyle='--', label="Static", alpha=0.8)
    ax5.set_xlabel("Time Slot")
    ax5.set_ylabel("Cumulative Energy (J)")
    ax5.set_title("Cumulative Energy Consumption")
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.3)

    # Offload Decisions (rolling average)
    window = 10
    rsplo_offload_ma = np.convolve(rsplo_u0["decision_history"], np.ones(window)/window, mode='valid')
    static_offload_ma = np.convolve(static_u0["decision_history"], np.ones(window)/window, mode='valid')
    ax6.plot(range(len(rsplo_offload_ma)), rsplo_offload_ma, color=PURPLE, linewidth=1.5, label="RS-PLO")
    ax6.plot(range(len(static_offload_ma)), static_offload_ma, color=YELLOW, linewidth=1.5, linestyle='--', label="Static", alpha=0.8)
    ax6.set_xlabel("Time Slot")
    ax6.set_ylabel("Offload Ratio (rolling avg)")
    ax6.set_title("Offloading Decision Pattern")
    ax6.set_ylim(-0.05, 1.05)
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.3)

    # Latency Comparison
    rsplo_lat_ma = np.convolve(np.array(rsplo_u0["latency_history"])*1000, np.ones(window)/window, mode='valid')
    static_lat_ma = np.convolve(np.array(static_u0["latency_history"])*1000, np.ones(window)/window, mode='valid')
    ax7.plot(range(len(rsplo_lat_ma)), rsplo_lat_ma, color=CYAN, linewidth=1.5, label="RS-PLO")
    ax7.plot(range(len(static_lat_ma)), static_lat_ma, color=RED, linewidth=1.5, linestyle='--', label="Static", alpha=0.8)
    ax7.set_xlabel("Time Slot")
    ax7.set_ylabel("Latency (ms)")
    ax7.set_title("Task Execution Latency")
    ax7.legend()
    ax7.grid(True, linestyle='--', alpha=0.3)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(os.path.join(output_dir, "02_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"       → Saved: {output_dir}/02_comparison.png")

    # ── FIGURE 3: Summary Bar Charts ──
    fig3, (ax8, ax9) = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle("Summary: RS-PLO vs Static Lyapunov", fontsize=16, fontweight='bold', y=1.02)

    # Average Queue by User
    x = np.arange(len(rsplo_results["users"]))
    width = 0.35
    rsplo_avg_q = [u["avg_Q"] for u in rsplo_results["users"]]
    static_avg_q = [u["avg_Q"] for u in static_results["users"]]
    bars1 = ax8.bar(x - width/2, rsplo_avg_q, width, color=CYAN, label="RS-PLO", alpha=0.85)
    bars2 = ax8.bar(x + width/2, static_avg_q, width, color=RED, label="Static", alpha=0.85)
    ax8.set_xlabel("User ID")
    ax8.set_ylabel("Average Queue Backlog")
    ax8.set_title("Per-User Average Queue Backlog")
    ax8.set_xticks(x)
    ax8.legend()
    ax8.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Overall Summary
    metrics = ["Avg Queue\n(bits)", "Avg Latency\n(ms)", "Offload\nRatio (%)"]
    rsplo_vals = [rsplo_results["avg_Q"], rsplo_results["avg_latency_ms"], rsplo_results["offload_ratio"]*100]
    static_vals = [static_results["avg_Q"], static_results["avg_latency_ms"], static_results["offload_ratio"]*100]

    # Normalize for visualization
    max_vals = [max(r, s) if max(r, s) > 0 else 1 for r, s in zip(rsplo_vals, static_vals)]
    rsplo_norm = [r/m for r, m in zip(rsplo_vals, max_vals)]
    static_norm = [s/m for s, m in zip(static_vals, max_vals)]

    x2 = np.arange(len(metrics))
    ax9.bar(x2 - width/2, rsplo_norm, width, color=GREEN, label="RS-PLO", alpha=0.85)
    ax9.bar(x2 + width/2, static_norm, width, color=RED, label="Static", alpha=0.85)

    # Add value labels
    for i, (rv, sv) in enumerate(zip(rsplo_vals, static_vals)):
        ax9.text(i - width/2, rsplo_norm[i] + 0.02, f"{rv:.1f}", ha='center', fontsize=9, color=GREEN)
        ax9.text(i + width/2, static_norm[i] + 0.02, f"{sv:.1f}", ha='center', fontsize=9, color=RED)

    ax9.set_ylabel("Normalized Value")
    ax9.set_title("Overall Performance Comparison")
    ax9.set_xticks(x2)
    ax9.set_xticklabels(metrics)
    ax9.legend()
    ax9.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "03_summary.png"), dpi=150, bbox_inches='tight')
    print(f"       → Saved: {output_dir}/03_summary.png")

    plt.close('all')


def print_summary(rsplo_results: dict, static_results: dict):
    """Print final summary statistics."""
    print()
    print("[5/5] Final Results")
    print("=" * 70)
    print(f"{'Metric':<30} {'RS-PLO':>15} {'Static':>15} {'Improvement':>15}")
    print("-" * 70)

    metrics = [
        ("Avg Queue Backlog (bits)", rsplo_results["avg_Q"], static_results["avg_Q"]),
        ("Avg Latency (ms)", rsplo_results["avg_latency_ms"], static_results["avg_latency_ms"]),
        ("Total Energy (J)", rsplo_results["total_energy"], static_results["total_energy"]),
        ("Offload Ratio (%)", rsplo_results["offload_ratio"]*100, static_results["offload_ratio"]*100),
    ]

    for name, rsplo_val, static_val in metrics:
        if static_val > 0:
            improvement = ((static_val - rsplo_val) / static_val) * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "N/A"
        print(f"{name:<30} {rsplo_val:>15.2f} {static_val:>15.2f} {imp_str:>15}")

    print("-" * 70)
    print()

    # Per-user breakdown
    print("Per-User Breakdown (RS-PLO):")
    print(f"{'User':<8} {'Avg Q':>12} {'Max Q':>12} {'Avg V':>10} {'Offload%':>10}")
    print("-" * 55)
    for u in rsplo_results["users"]:
        print(f"User {u['user_id']:<3} {u['avg_Q']:>12.1f} {u['max_Q']:>12.1f} "
              f"{u['avg_V']:>10.4f} {u['offload_ratio']*100:>9.1f}%")

    print()
    print("=" * 70)
    print("  KEY OBSERVATION: RS-PLO adapts V(t) based on channel volatility.")
    print("  When Z(t) spikes → V(t) drops → system prioritizes queue stability.")
    print("  Static baseline has no such mechanism and degrades under bursts.")
    print("=" * 70)


def main():
    """Main entry point — runs the full experiment."""

    # ── Configure system parameters ──
    params = SystemParams(
        N=5,                        # 5 IoT users
        V_max=10.0,                 # max control parameter
        beta=2.0,                   # V(t) sensitivity (higher = more aggressive response)
        gamma=0.2,                  # volatility decay rate (lower = slower risk forgetting)
        mean_task_size=50,          # base matrix/data size
        burst_probability=0.25,     # 25% chance of burst per slot
        burst_multiplier=3.0,       # 3x bigger tasks during bursts
        time_slots=200,             # 200 time slots
        edge_host="127.0.0.1",
        edge_port=9999,
    )

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    # ── Start edge servers (3 servers) ──
    server_ports = [s.port for s in params.edge_servers]
    server_procs = start_edge_servers(server_ports)

    try:
        # ── Run RS-PLO ──
        rsplo_results = run_rsplo(params)

        # ── Run Static Baseline ──
        static_results = run_static_baseline(params)

        # ── Generate Plots ──
        generate_plots(rsplo_results, static_results, output_dir)

        # ── Print Summary ──
        print_summary(rsplo_results, static_results)

        print(f"\nPlots saved to: {output_dir}/")
        print("Done! \u2713")

    finally:
        # ── Stop all edge servers ──
        for proc in server_procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"\n[EDGE SERVERS] All {len(server_procs)} servers stopped.")


if __name__ == "__main__":
    main()
