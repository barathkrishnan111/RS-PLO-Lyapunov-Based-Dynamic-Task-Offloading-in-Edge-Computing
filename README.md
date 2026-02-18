# ⚡ RS-PLO: Lyapunov-Based Dynamic Task Offloading in Edge Computing

> **Real-time, risk-sensitive task offloading using Lyapunov optimization — not a simulation, actual computation.**

This project implements the **RS-PLO (Risk-Sensitive Predictive Lyapunov Optimization)** framework for dynamic task offloading in Mobile Edge Computing (MEC) environments. Every task is **actually executed** — either on your local device or offloaded to a real TCP edge server.

Based on the paper: *"Lyapunov-Stable Dynamic Task Offloading in Non-Stationary Edge Environments: A Deterministic Drift-Plus-Penalty Framework"*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The RS-PLO Algorithm](#the-rs-plo-algorithm)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Live Interactive Demo](#1-live-interactive-demo-demopy)
  - [Automated Experiment](#2-automated-experiment-run_experimentpy)
  - [Standalone Edge Server](#3-standalone-edge-server)
- [Supported Task Types](#supported-task-types)
- [System Parameters](#system-parameters)
- [Results and Analysis](#results-and-analysis)
- [How the Decision Works](#how-the-decision-works)
- [Technical Details](#technical-details)

---

## Overview

In edge computing, IoT devices must decide whether to:
- **Execute tasks locally** (slow CPU, no network needed)
- **Offload to an edge server** (fast server, but costs transmission energy)

The challenge: **wireless channels are unpredictable**. A good channel can suddenly degrade (entering a tunnel, moving behind a building). Static offloading policies fail under this volatility.

**RS-PLO solves this** by dynamically adapting its decision-making based on real-time channel conditions:

| Channel Stable | Channel Volatile |
|:---:|:---:|
| V(t) = HIGH | V(t) = LOW |
| Optimize for energy savings | Prioritize queue stability |
| Offload when energy-efficient | Choose fastest option |

---

## Architecture

```
+------------------------+         TCP Socket          +------------------------+
|   IoT Device (You)     | ---- Offload Tasks ---->   |   Edge Server (MEC)    |
|                        |                              |                        |
|  - RS-PLO Engine       |         OR                   |  - Matrix Multiply     |
|  - Queue Manager       |                              |  - SHA-256 Hashing     |
|  - Channel Monitor     |  <--- Execute Locally        |  - Prime Factorize     |
|  - Decision Maker      |                              |  - Word Count          |
|                        |  <--- Return Results ----    |  - File Analysis       |
+------------------------+                              |  - Text Encryption     |
   lyapunov_engine.py                                   |  - Number Sorting      |
   demo.py                                              +------------------------+
                                                           edge_server.py
```

**Key point:** The edge server runs as a **separate process** on `127.0.0.1:9999`. Tasks are sent over **real TCP sockets** with a length-prefixed JSON protocol. This mirrors how real MEC offloading works.

---

## The RS-PLO Algorithm

### Three Coupled Queues

```
                    Channel Prediction
                    Error e(t)
                         |
                         v
  Tasks --> [ Physical Queue Q(t) ]      [ Volatility Queue Z(t) ] --> V(t) = V_max * e^(-B*Z)
              |                              |
              |  Q grows when               |  Z grows when channel
              |  tasks arrive faster         |  is unpredictable
              |  than they're served         |
              v                              v
         Drift Term                     Penalty Term
         Q(t)*(A - u)                   V(t) * E(t)
              |                              |
              +--------- Cost Function ------+
                         |
                         v
                   DECISION: LOCAL or OFFLOAD
                   (pick the lower cost)
```

### Mathematical Formulation

**Per-slot optimization — minimize:**

```
Cost(x) = Q(t) * [A(t) - u_x(t)]  +  V(t) * E_x(t)
           |---- Drift Term ----|     |-- Penalty --|
           Queue growth pressure      Energy cost
```

Where:

| Symbol | Description |
|--------|-------------|
| `Q(t)` | Physical queue backlog (bits) — how many tasks are waiting |
| `A(t)` | Task arrival (bits this slot) |
| `u_x(t)` | Service rate — local CPU (10 MHz) or wireless TX rate |
| `V(t)` | Adaptive control = `V_max * exp(-beta * Z(t))` |
| `E_x(t)` | Energy cost of executing locally or offloading |
| `Z(t)` | Volatility queue = `max(Z(t-1) - gamma, 0) + e(t)` |
| `e(t)` | Channel prediction error (dB scale) |

### Key Behavior

| Condition | Z(t) | V(t) | System Behavior |
|-----------|:-----:|:-----:|-----------------|
| Stable channel | Low | **High** | Energy penalty dominates, choose cheaper energy option |
| Volatile channel | High | **Low** | Drift dominates, choose fastest service rate (clear queue) |
| Full queue + good channel | High Q | Any | Offload to edge (higher service rate) |
| Full queue + bad channel | High Q | Low | Execute locally (edge too slow over bad channel) |

---

## Project Structure

```
FP/
|-- edge_server.py          # TCP edge server - executes real computation tasks
|-- lyapunov_engine.py      # RS-PLO algorithm engine + static baseline
|-- demo.py                 # Live interactive CLI - process YOUR data in real-time
|-- run_experiment.py       # Automated experiment - RS-PLO vs Static comparison
|-- README.md               # This file
|-- paper_text.txt          # Extracted paper text
+-- results/                # Generated comparison plots
    |-- 01_rsplo_mechanism.png
    |-- 02_comparison.png
    +-- 03_summary.png
```

### File Details

| File | Lines | Purpose |
|------|------:|---------|
| `edge_server.py` | ~300 | TCP server with 7 task handlers, threaded client connections |
| `lyapunov_engine.py` | ~500 | RS-PLO + Static Lyapunov engines, channel model, mobility |
| `demo.py` | ~430 | Interactive CLI with live RS-PLO decisions on real tasks |
| `run_experiment.py` | ~350 | Orchestrates full experiment, generates Matplotlib plots |

---

## Installation

### Prerequisites

- **Python 3.8+** (tested with 3.13.7)
- **numpy** — numerical computation
- **matplotlib** — plot generation (only for `run_experiment.py`)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/barathkrishnan111/RS-PLO-Lyapunov-Based-Dynamic-Task-Offloading-in-Edge-Computing.git
cd RS-PLO-Lyapunov-Based-Dynamic-Task-Offloading-in-Edge-Computing

# 2. Install dependencies
pip install numpy matplotlib
```

No other setup needed. The edge server starts automatically.

---

## Usage

### 1. Live Interactive Demo (`demo.py`)

**This is the main way to use the system.** Start it and type commands in real-time:

```bash
python demo.py
```

#### Compute Tasks

```
rsplo> matrix 100          # Multiply two 100x100 matrices
rsplo> hash 500            # SHA-256 hash 500KB of data
rsplo> prime 9999991       # Factorize a large number
rsplo> sort 100000         # Sort 100,000 random numbers
rsplo> encrypt Hello World # Caesar cipher encrypt text
```

#### Process Your Own Files

```
rsplo> file path/to/your/file.txt    # Word count + entropy analysis
rsplo> hashfile path/to/any/file     # SHA-256 hash any file
```

#### Change Network Conditions (live!)

```
rsplo> move close          # 100m from edge (excellent signal)
rsplo> move far            # 2000m from edge (poor signal)
rsplo> tunnel              # Enter tunnel (very weak signal)
rsplo> distance 500        # Set exact distance in meters
```

#### System Commands

```
rsplo> status              # Show Q(t), Z(t), V(t), channel quality
rsplo> burst               # Send 5 rapid tasks (stress test)
rsplo> history             # Show full task execution log
rsplo> help                # List all commands
rsplo> quit                # Exit
```

#### Example Session

```
rsplo> matrix 100
  +--- Task #1: Matrix 100x100 ---
  |  Channel: -89.5dB | Distance: 300m | TX Rate: 27.2 Mbps
  |  Q=0 | Z=0.000 | V=10.0000
  |  Decision: OFFLOAD -> Edge Server
  |  Done! Executed on EDGE in 21.16ms

rsplo> tunnel
  ENTERED TUNNEL - Very weak signal! Distance: 2400m

rsplo> encrypt Secret message
  +--- Task #2: Encrypt text (14 chars) ---
  |  Channel: -118.9dB | Distance: 2400m | TX Rate: 0.1 Mbps
  |  Q=640,000 | Z=6.442 | V=0.0000
  |  Decision: LOCAL -> This Device
  |  Done! Executed on LOCAL in 0.00ms

rsplo> move close
  Moved CLOSE to edge server: 100m (excellent signal)

rsplo> matrix 80
  +--- Task #3: Matrix 80x80 ---
  |  Channel: -84.1dB | Distance: 100m | TX Rate: 43.5 Mbps
  |  Decision: OFFLOAD -> Edge Server
  |  Done! Executed on EDGE in 1.14ms
```

Notice how the **same task type** gets different decisions based on channel conditions!

---

### 2. Automated Experiment (`run_experiment.py`)

Runs a full comparison between RS-PLO (adaptive) and Static Lyapunov (fixed V):

```bash
python run_experiment.py
```

**What it does:**
1. Starts the edge server
2. Runs RS-PLO with 5 users over 200 time slots
3. Runs Static baseline with the same conditions
4. Generates 3 comparison plots in `results/`
5. Prints summary statistics

**Output plots:**

| Plot | Shows |
|------|-------|
| `01_rsplo_mechanism.png` | Q(t), Z(t), V(t) over time - the adaptive mechanism |
| `02_comparison.png` | RS-PLO vs Static: queue, energy, offload pattern, latency |
| `03_summary.png` | Per-user comparison and overall metrics |

---

### 3. Standalone Edge Server

You can run the edge server independently and send tasks from your own code:

```bash
# Terminal 1: Start the server
python edge_server.py 9999
```

```python
# Terminal 2: Send a task (Python)
import socket, json, struct

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 9999))

# Send task
task = {"task_type": "matrix_multiply", "params": {"size": 100}, "task_id": 1}
payload = json.dumps(task).encode()
sock.sendall(struct.pack("!I", len(payload)) + payload)

# Receive result
raw_len = sock.recv(4)
msg_len = struct.unpack("!I", raw_len)[0]
result = json.loads(sock.recv(msg_len).decode())
print(result)
sock.close()
```

---

## Supported Task Types

| Task Type | Command | What It Computes | Data Size |
|-----------|---------|-----------------|-----------|
| `matrix_multiply` | `matrix N` | NxN matrix multiplication (real NumPy dot) | N^2 x 64 bits |
| `hash_data` | `hash KB` | SHA-256 with 100 chained rounds | KB x 8000 bits |
| `prime_factorize` | `prime N` | Trial-division prime factorization | ~20K bits |
| `sort_numbers` | `sort count` | Sort N random numbers | N x 64 bits |
| `text_encrypt` | `encrypt text` | Caesar cipher + SHA-256 hash | len x 8 bits |
| `word_count` | `file path` | Word/line/char count + frequency | file size |
| `file_stats` | `file path` | Shannon entropy + pattern analysis | file size |

All tasks are **real computation** — actual matrix products, actual hashing, actual factorization.

---

## System Parameters

Configurable in `lyapunov_engine.py` (`SystemParams` class) or in `run_experiment.py`:

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `V_max` | 10.0 | Maximum control parameter (energy weight when stable) |
| `beta` | 2.0 | Sensitivity of V(t) to volatility (higher = faster response) |
| `gamma` | 0.2 | Volatility queue decay rate (lower = longer memory of risk) |

### Device Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `f_local` | 10 MHz | Local IoT CPU frequency (microcontroller) |
| `f_edge` | 5 GHz | Edge server CPU frequency |
| `P_tx` | 0.5 W | Wireless transmission power |
| `local_energy_per_bit` | 5e-6 J/bit | Local computation energy cost |

### Channel Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `h0` | 1e-2 | Reference channel gain at 1m |
| `alpha` | 3.0 | Path loss exponent (urban micro-cell) |
| `bandwidth` | 10 MHz | Wireless channel bandwidth |
| `noise_power` | 1e-10 | Noise power spectral density |

### Scenario Presets

```python
# Stable office WiFi
params = SystemParams(burst_probability=0.05, gamma=0.5)

# Moving vehicle (high volatility)
params = SystemParams(burst_probability=0.3, beta=3.0, gamma=0.1)

# Factory IoT flood
params = SystemParams(N=10, burst_probability=0.5, burst_multiplier=5.0)

# Energy-constrained sensor
params = SystemParams(V_max=20.0)  # prioritize energy saving

# Latency-critical drone
params = SystemParams(V_max=2.0)   # prioritize queue clearing
```

---

## Results and Analysis

### RS-PLO vs Static Lyapunov

| Metric | RS-PLO | Static | Difference |
|--------|:------:|:------:|:----------:|
| Avg Latency | 0.26 ms | 0.31 ms | **15.3% better** |
| Offload Ratio | 4.3% | 8.0% | RS-PLO is more selective |
| Total Energy | 1805 J | 1663 J | Static saves energy blindly |
| Avg Queue | 37.85M bits | 37.85M bits | Comparable stability |

**Key insight:** RS-PLO offloads **less** but achieves **lower latency** because it only offloads when conditions are favorable. Static Lyapunov always weights energy the same, causing it to offload even in suboptimal conditions.

### How V(t) Adapts

```
Time -->  Stable Channel    |    Tunnel Event    |    Recovery
          ----------------------------------------------------------
Z(t):     0.0  0.1  0.2    |   5.0  8.0  12.0  |   11.8  11.6
V(t):     10.0  9.8  9.6   |   0.0  0.0  0.0   |   0.0   0.0
Mode:     Energy-optimize   |   Queue-stabilize  |   Recovering
```

When Z(t) is high, V(t) drops to near-zero, making the system ignore energy costs and focus purely on clearing the queue as fast as possible.

---

## How the Decision Works

For each incoming task, the algorithm computes:

```
Cost_local   = Q_norm x drift_local   + V(t) x penalty_local
Cost_offload = Q_norm x drift_offload + V(t) x penalty_offload
```

**Pick the option with lower cost.**

The drift captures **queue pressure** (how fast each option clears the queue), and the penalty captures **energy cost**. V(t) controls the balance:

- **V = 10** (stable): Energy matters a lot - pick the cheaper option
- **V = 0** (volatile): Energy does not matter - pick the fastest option
- **Large Q**: Queue pressure dominates regardless - drain the queue

This creates an **intelligent, adaptive** offloading policy that responds to real-time conditions — exactly what the paper proposes.

---

## Technical Details

### Channel Model
- Path loss: `h(d) = h0 * d^(-alpha)` with Rayleigh fading
- Shannon capacity: `R = B * log2(1 + P*h/N0)`
- Prediction error computed in **dB scale** for meaningful volatility tracking

### Communication Protocol
- TCP sockets with length-prefixed JSON (4-byte big-endian header)
- Threaded server handles multiple concurrent clients
- Automatic fallback to local execution if edge connection fails

### Energy Model
- Local: `E_local = energy_per_bit * task_bits` (slow CPU = more energy per bit)
- Offload: `E_offload = P_tx * (task_bits / R)` (depends on channel quality)
- When channel is good: offload is cheaper. When bad: local is cheaper.

---

## License

Academic/research use. Based on the RS-PLO framework from the referenced paper.