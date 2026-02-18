"""
LIVE Interactive RS-PLO Demo — Real-Time Task Offloading

This is NOT a simulation. Every task you submit is ACTUALLY:
  - Executed on your LOCAL machine, OR
  - Sent over TCP to the EDGE SERVER and executed there

The RS-PLO algorithm makes the decision in real-time based on:
  - Current queue backlog Q(t)
  - Channel volatility Z(t)
  - Adaptive control V(t)

Usage:  python demo.py
"""

import subprocess
import sys
import os
import time
import socket
import struct
import json
import hashlib
import threading
import numpy as np

# ─── Add project root to path ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lyapunov_engine import RSPLO, SystemParams, execute_locally, execute_on_edge


# ─────────────────────────────────────────────
#  COLORS for terminal output
# ─────────────────────────────────────────────
class C:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"
    WHITE   = "\033[97m"
    BG_BLUE = "\033[44m"


# ─────────────────────────────────────────────
#  LIVE DEMO ENGINE
# ─────────────────────────────────────────────

class LiveDemo:
    """
    Interactive, LIVE RS-PLO task offloading system.
    Every task is actually computed — either locally or on the edge server.
    """

    def __init__(self, port=9999):
        self.port = port
        self.server_proc = None
        self.task_count = 0

        # RS-PLO engine with a single user (you!)
        self.params = SystemParams(
            N=1,
            V_max=10.0,
            beta=2.0,
            gamma=0.2,
            mean_task_size=50,
            burst_probability=0.0,  # no auto-bursts, you control everything
            burst_multiplier=1.0,
            time_slots=9999,
            edge_host="127.0.0.1",
            edge_port=port,
        )
        self.engine = RSPLO(self.params, seed=42)
        self.user = self.engine.users[0]
        self.user.distance = 300  # start 300m from edge server

        # History for live display
        self.task_log = []

    def start_server(self):
        """Start the edge server as a background process."""
        self.server_proc = subprocess.Popen(
            [sys.executable, "edge_server.py", str(self.port)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        time.sleep(1.5)
        if self.server_proc.poll() is not None:
            print(f"{C.RED}[ERROR] Edge server failed to start!{C.RESET}")
            sys.exit(1)

    def stop_server(self):
        """Stop the edge server."""
        if self.server_proc:
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()

    def show_banner(self):
        print()
        print(f"{C.BOLD}{C.CYAN}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}{C.WHITE}  ⚡ LIVE RS-PLO TASK OFFLOADING — Real-Time Edge Computing{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}{'═'*70}{C.RESET}")
        print(f"{C.DIM}  Edge Server: 127.0.0.1:{self.port} | Your Device: IoT MCU (10 MHz){C.RESET}")
        print(f"{C.DIM}  Every task is ACTUALLY executed — not simulated!{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}{'═'*70}{C.RESET}")
        print()

    def show_help(self):
        print(f"""
{C.BOLD}{C.WHITE}  Available Commands:{C.RESET}
{C.CYAN}  ────────────────────────────────────────────────────{C.RESET}

  {C.GREEN}COMPUTE TASKS:{C.RESET}
    {C.YELLOW}matrix <size>{C.RESET}          Multiply two NxN matrices (e.g. matrix 100)
    {C.YELLOW}hash <size_kb>{C.RESET}         SHA-256 hash data (e.g. hash 500)
    {C.YELLOW}prime <number>{C.RESET}         Prime factorize a number (e.g. prime 9999991)
    {C.YELLOW}sort <count>{C.RESET}           Sort N random numbers (e.g. sort 100000)
    {C.YELLOW}encrypt <text>{C.RESET}         Caesar cipher encrypt text (e.g. encrypt Hello World)

  {C.GREEN}FILE PROCESSING:{C.RESET}
    {C.YELLOW}file <path>{C.RESET}            Process a file — word count + entropy analysis
    {C.YELLOW}hashfile <path>{C.RESET}        SHA-256 hash a file's contents

  {C.GREEN}NETWORK CONDITIONS:{C.RESET}
    {C.YELLOW}distance <meters>{C.RESET}      Set distance to edge server (30-2500m)
    {C.YELLOW}move close{C.RESET}             Move to 100m (excellent signal)
    {C.YELLOW}move far{C.RESET}               Move to 2000m (poor signal)
    {C.YELLOW}tunnel{C.RESET}                 Enter a tunnel (very poor signal)

  {C.GREEN}SYSTEM:{C.RESET}
    {C.YELLOW}status{C.RESET}                 Show current RS-PLO state (Q, Z, V, channel)
    {C.YELLOW}burst{C.RESET}                  Send 5 rapid tasks (stress test)
    {C.YELLOW}history{C.RESET}                Show task execution history
    {C.YELLOW}help{C.RESET}                   Show this help
    {C.YELLOW}quit{C.RESET}                   Exit

{C.CYAN}  ────────────────────────────────────────────────────{C.RESET}
""")

    def show_status(self):
        """Show current RS-PLO state."""
        channel_gain = self.engine.compute_channel_gain(self.user)
        channel_db = self.engine.compute_channel_gain_db(channel_gain)
        V_t = self.engine.compute_V(self.user.Z)
        tx_rate = self.engine.compute_transmission_rate(channel_gain)

        # Signal strength bars
        if channel_db > -90:
            signal = f"{C.GREEN}████{C.RESET} Excellent"
        elif channel_db > -100:
            signal = f"{C.GREEN}███{C.DIM}█{C.RESET} Good"
        elif channel_db > -110:
            signal = f"{C.YELLOW}██{C.DIM}██{C.RESET} Fair"
        elif channel_db > -120:
            signal = f"{C.RED}█{C.DIM}███{C.RESET} Poor"
        else:
            signal = f"{C.RED}{C.DIM}████{C.RESET} Very Poor"

        print(f"""
{C.BOLD}{C.WHITE}  ┌─── RS-PLO State ───────────────────────────────┐{C.RESET}
  │  {C.CYAN}Queue Backlog Q(t):{C.RESET}  {self.user.Q:>15,.0f} bits        │
  │  {C.RED}Volatility Z(t):  {C.RESET}  {self.user.Z:>15.3f}              │
  │  {C.GREEN}Control V(t):     {C.RESET}  {V_t:>15.4f}              │
  │  {C.YELLOW}Distance:         {C.RESET}  {self.user.distance:>12,.0f} m            │
  │  {C.MAGENTA}Channel Gain:     {C.RESET}  {channel_db:>12.1f} dB           │
  │  {C.WHITE}Signal:           {C.RESET}  {signal}                  │
  │  {C.WHITE}TX Rate:          {C.RESET}  {tx_rate/1e6:>12.1f} Mbps         │
  │  {C.WHITE}Tasks Processed:  {C.RESET}  {self.task_count:>12}              │
{C.BOLD}{C.WHITE}  └─────────────────────────────────────────────────┘{C.RESET}
""")

    def process_task(self, task_type: str, params: dict, task_bits: int, label: str):
        """
        Run one task through the RS-PLO decision engine — LIVE.
        The task is ACTUALLY executed locally or on the edge server.
        """
        self.task_count += 1

        # 1. Update mobility (small random walk from current position)
        delta = self.engine.rng.normal(0, 5)
        self.user.distance = np.clip(self.user.distance + delta, 30, 2500)

        # 2. Compute channel
        channel_gain = self.engine.compute_channel_gain(self.user)
        channel_db = self.engine.compute_channel_gain_db(channel_gain)

        # 3. Update volatility
        if self.user.prev_channel_gain > 0:
            prev_db = self.engine.compute_channel_gain_db(self.user.prev_channel_gain)
            prediction_error = abs(channel_db - prev_db) / 10.0
        else:
            prediction_error = 0.0
        self.user.prev_channel_gain = channel_gain
        self.user.Z = max(self.user.Z - self.params.gamma, 0) + prediction_error

        # 4. Compute V(t)
        V_t = self.engine.compute_V(self.user.Z)

        # 5. RS-PLO decision
        task_obj = {"task_type": task_type, "params": params, "task_bits": task_bits}
        decision = self.engine.drift_plus_penalty_decision(self.user, task_obj, channel_gain)

        # 6. Print decision reasoning
        tx_rate = self.engine.compute_transmission_rate(channel_gain)
        local_energy = self.engine.compute_local_energy(task_bits)
        offload_energy = self.engine.compute_offload_energy(task_bits, tx_rate)

        print(f"\n{C.BOLD}{C.WHITE}  ┌─── Task #{self.task_count}: {label} ───{C.RESET}")
        print(f"  │  {C.DIM}Channel: {channel_db:.1f}dB | Distance: {self.user.distance:.0f}m | TX Rate: {tx_rate/1e6:.1f} Mbps{C.RESET}")
        print(f"  │  {C.DIM}Q={self.user.Q:,.0f} | Z={self.user.Z:.3f} | V={V_t:.4f}{C.RESET}")
        print(f"  │  {C.DIM}Energy — Local: {local_energy*1000:.2f}mJ | Offload: {offload_energy*1000:.2f}mJ{C.RESET}")

        if decision == 1:
            dec_str = f"{C.CYAN}⚡ OFFLOAD → Edge Server{C.RESET}"
        else:
            dec_str = f"{C.YELLOW}🏠 LOCAL → This Device{C.RESET}"
        print(f"  │  {C.BOLD}Decision: {dec_str}")

        # 7. ACTUALLY EXECUTE
        print(f"  │  {C.DIM}Executing...{C.RESET}", end="", flush=True)
        start = time.perf_counter()

        if decision == 1:
            result = execute_on_edge(task_type, params, self.task_count,
                                     self.params.edge_host, self.params.edge_port)
            if result.get("error"):
                print(f"\r  │  {C.RED}Edge failed! Falling back to local...{C.RESET}")
                result = execute_locally(task_type, params)
                decision = 0
        else:
            result = execute_locally(task_type, params)

        elapsed = time.perf_counter() - start
        exec_ms = result["exec_time"] * 1000

        # 8. Update queue
        if decision == 1:
            service_bits = tx_rate * result["exec_time"]
        else:
            service_bits = self.params.f_local * result["exec_time"]
        self.user.Q = max(self.user.Q - service_bits, 0) + task_bits

        # 9. Record
        energy = offload_energy if decision == 1 else local_energy
        self.user.Q_history.append(self.user.Q)
        self.user.Z_history.append(self.user.Z)
        self.user.V_history.append(V_t)
        self.user.energy_history.append(energy)
        self.user.decision_history.append(decision)
        self.user.latency_history.append(result["exec_time"])

        self.task_log.append({
            "id": self.task_count,
            "label": label,
            "decision": "EDGE" if decision == 1 else "LOCAL",
            "exec_ms": exec_ms,
            "energy_mJ": energy * 1000,
            "Q": self.user.Q,
            "Z": self.user.Z,
            "V": V_t,
            "channel_db": channel_db,
        })

        # 10. Show results
        location = f"{C.CYAN}EDGE{C.RESET}" if decision == 1 else f"{C.YELLOW}LOCAL{C.RESET}"
        print(f"\r  │  {C.GREEN}✓ Done!{C.RESET} Executed on {location} in {C.BOLD}{exec_ms:.2f}ms{C.RESET}")

        # Show task-specific results
        res = result.get("result", result)
        if task_type == "matrix_multiply":
            checksum = res.get("checksum", res.get("result_checksum", "?"))
            print(f"  │  {C.DIM}Result: checksum = {checksum}{C.RESET}")
        elif task_type == "hash_data":
            h = res.get("hash", res.get("result_hash", "?"))
            print(f"  │  {C.DIM}Result: SHA-256 = {h}...{C.RESET}")
        elif task_type == "prime_factorize":
            fc = res.get("factors_count", "?")
            lg = res.get("largest", res.get("largest_factor", "?"))
            print(f"  │  {C.DIM}Result: {fc} factors, largest = {lg}{C.RESET}")
        elif task_type == "word_count":
            wc = res.get("word_count", "?")
            lc = res.get("line_count", "?")
            uw = res.get("unique_words", "?")
            top = res.get("top_10_words", [])
            print(f"  │  {C.DIM}Result: {wc} words, {lc} lines, {uw} unique words{C.RESET}")
            if top:
                top_str = ", ".join([f"{w}({c})" for w, c in top[:5]])
                print(f"  │  {C.DIM}Top words: {top_str}{C.RESET}")
        elif task_type == "sort_numbers":
            print(f"  │  {C.DIM}Result: min={res.get('min')}, max={res.get('max')}, median={res.get('median')}, mean={res.get('mean', 0):.2f}{C.RESET}")
        elif task_type == "text_encrypt":
            enc = res.get("encrypted_preview", "?")
            sha = res.get("sha256", "?")
            print(f"  │  {C.DIM}Encrypted: {enc[:80]}...{C.RESET}")
            print(f"  │  {C.DIM}SHA-256: {sha}{C.RESET}")
        elif task_type == "file_stats":
            ent = res.get("entropy_bits", "?")
            sz = res.get("size_bytes", "?")
            print(f"  │  {C.DIM}Result: {sz} bytes, entropy = {ent} bits/char{C.RESET}")

        print(f"  │  {C.DIM}Queue: {self.user.Q:,.0f} bits | Energy: {energy*1000:.2f}mJ{C.RESET}")
        print(f"{C.BOLD}{C.WHITE}  └────────────────────────────────────────────────{C.RESET}")

    def show_history(self):
        """Show task execution history."""
        if not self.task_log:
            print(f"  {C.DIM}No tasks executed yet.{C.RESET}")
            return
        print(f"\n{C.BOLD}{C.WHITE}  Task History:{C.RESET}")
        print(f"  {'#':>3} {'Task':<25} {'Where':>6} {'Time':>10} {'Energy':>10} {'Q':>14} {'Z':>8} {'V':>8}")
        print(f"  {'-'*90}")
        for t in self.task_log:
            color = C.CYAN if t["decision"] == "EDGE" else C.YELLOW
            print(f"  {t['id']:>3} {t['label']:<25} {color}{t['decision']:>6}{C.RESET} "
                  f"{t['exec_ms']:>8.2f}ms {t['energy_mJ']:>8.2f}mJ "
                  f"{t['Q']:>14,.0f} {t['Z']:>8.3f} {t['V']:>8.4f}")
        print()
        offloaded = sum(1 for t in self.task_log if t["decision"] == "EDGE")
        print(f"  {C.DIM}Total: {len(self.task_log)} tasks | "
              f"Offloaded: {offloaded} ({offloaded/len(self.task_log)*100:.0f}%) | "
              f"Local: {len(self.task_log)-offloaded} ({(len(self.task_log)-offloaded)/len(self.task_log)*100:.0f}%){C.RESET}")
        print()

    def handle_command(self, raw: str):
        """Parse and execute a user command."""
        parts = raw.strip().split(None, 1)
        if not parts:
            return True
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # ── COMPUTE TASKS ──
        if cmd == "matrix":
            size = int(arg) if arg.isdigit() else 50
            task_bits = size * size * 64
            self.process_task("matrix_multiply", {"size": size}, task_bits, f"Matrix {size}x{size}")

        elif cmd == "hash":
            kb = int(arg) if arg.isdigit() else 100
            size_bytes = kb * 1024
            task_bits = size_bytes * 8
            self.process_task("hash_data", {"size_bytes": size_bytes}, task_bits, f"SHA-256 {kb}KB")

        elif cmd == "prime":
            n = int(arg) if arg.isdigit() else 999983
            task_bits = n.bit_length() * 1000
            self.process_task("prime_factorize", {"n": n}, task_bits, f"Factorize {n}")

        elif cmd == "sort":
            count = int(arg) if arg.isdigit() else 10000
            numbers = list(np.random.rand(count) * 1000000)
            task_bits = count * 64
            self.process_task("sort_numbers", {"numbers": numbers}, task_bits, f"Sort {count} nums")

        elif cmd == "encrypt":
            text = arg if arg else "Hello World from RS-PLO!"
            task_bits = len(text.encode()) * 8
            self.process_task("text_encrypt", {"text": text, "shift": 13}, task_bits, f"Encrypt text ({len(text)} chars)")

        # ── FILE PROCESSING ──
        elif cmd == "file":
            path = arg.strip().strip('"').strip("'")
            if not path:
                print(f"  {C.RED}Usage: file <path>{C.RESET}")
                return True
            if not os.path.isfile(path):
                print(f"  {C.RED}File not found: {path}{C.RESET}")
                return True
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                task_bits = len(text.encode()) * 8
                print(f"  {C.DIM}Read {len(text)} characters from {os.path.basename(path)}{C.RESET}")

                # Run BOTH word_count and file_stats
                self.process_task("word_count", {"text": text}, task_bits,
                                  f"Word count: {os.path.basename(path)}")
                self.process_task("file_stats", {"data": text}, task_bits,
                                  f"File stats: {os.path.basename(path)}")
            except Exception as e:
                print(f"  {C.RED}Error reading file: {e}{C.RESET}")

        elif cmd == "hashfile":
            path = arg.strip().strip('"').strip("'")
            if not path:
                print(f"  {C.RED}Usage: hashfile <path>{C.RESET}")
                return True
            if not os.path.isfile(path):
                print(f"  {C.RED}File not found: {path}{C.RESET}")
                return True
            try:
                with open(path, "rb") as f:
                    data = f.read()
                task_bits = len(data) * 8
                print(f"  {C.DIM}Read {len(data)} bytes from {os.path.basename(path)}{C.RESET}")
                self.process_task("hash_data", {"size_bytes": len(data)}, task_bits,
                                  f"SHA-256: {os.path.basename(path)}")
            except Exception as e:
                print(f"  {C.RED}Error reading file: {e}{C.RESET}")

        # ── NETWORK CONDITIONS ──
        elif cmd == "distance":
            d = float(arg) if arg else 300
            d = max(30, min(2500, d))
            old = self.user.distance
            self.user.distance = d
            print(f"  {C.MAGENTA}📡 Moved: {old:.0f}m → {d:.0f}m from edge server{C.RESET}")
            self.show_status()

        elif cmd == "move":
            if arg.lower() == "close":
                self.user.distance = 100
                print(f"  {C.GREEN}📡 Moved CLOSE to edge server: 100m (excellent signal){C.RESET}")
            elif arg.lower() == "far":
                self.user.distance = 2000
                print(f"  {C.RED}📡 Moved FAR from edge server: 2000m (poor signal){C.RESET}")
            else:
                print(f"  {C.RED}Usage: move close / move far{C.RESET}")
                return True
            self.show_status()

        elif cmd == "tunnel":
            self.user.distance = 2400
            print(f"  {C.RED}🚇 ENTERED TUNNEL — Very weak signal! Distance: 2400m{C.RESET}")
            # Spike the volatility
            self.user.Z += 5.0
            self.show_status()

        # ── SYSTEM ──
        elif cmd == "status":
            self.show_status()

        elif cmd == "burst":
            print(f"  {C.MAGENTA}🔥 BURST MODE — Sending 5 rapid tasks...{C.RESET}")
            tasks = [
                ("matrix", "80"),
                ("hash", "200"),
                ("prime", "9999991"),
                ("sort", "50000"),
                ("matrix", "120"),
            ]
            for t_cmd, t_arg in tasks:
                self.handle_command(f"{t_cmd} {t_arg}")

        elif cmd == "history":
            self.show_history()

        elif cmd == "help":
            self.show_help()

        elif cmd in ("quit", "exit", "q"):
            return False

        else:
            print(f"  {C.RED}Unknown command: {cmd}. Type 'help' for commands.{C.RESET}")

        return True

    def run(self):
        """Main interactive loop."""
        self.start_server()
        self.show_banner()
        self.show_help()
        self.show_status()

        print(f"{C.GREEN}  Ready! Type a command (or 'help'):{C.RESET}")

        running = True
        while running:
            try:
                prompt = f"{C.BOLD}{C.CYAN}  rsplo>{C.RESET} "
                raw = input(prompt)
                running = self.handle_command(raw)
            except (KeyboardInterrupt, EOFError):
                print()
                running = False

        print(f"\n{C.DIM}  Shutting down edge server...{C.RESET}")
        self.stop_server()

        if self.task_log:
            self.show_history()
        print(f"  {C.GREEN}Goodbye! ✓{C.RESET}\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo = LiveDemo()
    demo.run()
