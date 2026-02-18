# RS-PLO: Lyapunov-Based Dynamic Task Offloading in Edge Computing

> **Real-time, risk-sensitive task offloading using Lyapunov optimization -- not a simulation, actual computation.**

This project implements the **RS-PLO (Risk-Sensitive Predictive Lyapunov Optimization)** framework for dynamic task offloading in Mobile Edge Computing (MEC) environments. Every task is **actually executed** -- either on your local device or offloaded to one of **3 real TCP edge servers** at different distances.

Based on the paper: *"Lyapunov-Stable Dynamic Task Offloading in Non-Stationary Edge Environments: A Deterministic Drift-Plus-Penalty Framework"*

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The RS-PLO Algorithm](#the-rs-plo-algorithm)
- [Multi-Edge Server Selection](#multi-edge-server-selection)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Live Dashboard](#1-live-dashboard-dashboardpy)
  - [Interactive CLI Demo](#2-interactive-cli-demo-demopy)
  - [Automated Experiment](#3-automated-experiment-run_experimentpy)
  - [Standalone Edge Server](#4-standalone-edge-server)
- [Supported Task Types](#supported-task-types)
- [System Parameters](#system-parameters)
- [Results and Analysis](#results-and-analysis)
- [How the Decision Works](#how-the-decision-works)
- [Technical Details](#technical-details)
- [Docker](#docker)

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
                          Multi-Edge Environment
                    +-------------------------------+
+----------------+  |  +----------+  +-----------+  |
|  IoT Device    |---->| Edge-1   |  | Edge-2    |  |
|                |  |  | (Near)   |  | (Mid)     |  |
|  RS-PLO Engine |  |  | 150m     |  | 600m      |  |
|  Queue Manager |  |  | Port 9999|  | Port 10000|  |
|  Multi-Edge DPP|  |  +----------+  +-----------+  |
|                |  |                                |
|  OR: Execute   |  |  +----------+                  |
|  Locally       |  |  | Edge-3   |                  |
+----------------+  |  | (Far)    |                  |
  lyapunov_engine   |  | 1200m    |                  |
  demo.py           |  | Port 10001                  |
  dashboard.py      |  +----------+                  |
                    +-------------------------------+
                          edge_server.py (x3)
```

**Key points:**
- **3 edge servers** at different distances (150m, 600m, 1200m) with varying compute power
- Tasks sent over **real TCP sockets** with length-prefixed JSON protocol
- The RS-PLO algorithm evaluates **4 options per task** (LOCAL + 3 edge servers)
- A **live web dashboard** at `http://localhost:8080` shows real-time charts

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
                   DECISION: LOCAL or EDGE-1 or EDGE-2 or EDGE-3
                   (pick the lowest cost among all 4 options)
```

### Mathematical Formulation

**Per-slot optimization -- minimize for each option x:**

```
Cost(x) = Q(t) * [A(t) - u_x(t)]  +  V(t) * E_x(t)
           |---- Drift Term ----|     |-- Penalty --|
           Queue growth pressure      Energy cost
```

Where:

| Symbol | Description |
|--------|-------------|
| `Q(t)` | Physical queue backlog (bits) -- how many tasks are waiting |
| `A(t)` | Task arrival (bits this slot) |
| `u_x(t)` | Service rate -- local CPU or wireless TX rate to server x |
| `V(t)` | Adaptive control = `V_max * exp(-beta * Z(t))` |
| `E_x(t)` | Energy cost of executing on option x |
| `Z(t)` | Volatility queue = `max(Z(t-1) - gamma, 0) + e(t)` |
| `e(t)` | Channel prediction error (dB scale) |

---

## Multi-Edge Server Selection

The system evaluates **all options** per task and picks the minimum cost:

| Option | Distance | Port | Compute | When Chosen |
|--------|----------|------|:-------:|-------------|
| LOCAL | -- | -- | 1x | Bad channel, high volatility |
| Edge-1 (Near) | 150m | 9999 | 1.0x | Close range, standard tasks |
| Edge-2 (Mid) | 600m | 10000 | 1.5x | Medium range, powerful CPU |
| Edge-3 (Far) | 1200m | 10001 | 2.0x | Far range, GPU-class compute |

Each server has different channel conditions based on distance, and the DPP cost function evaluates them all:

```
Cost_local  = Q_norm * drift_local  + V(t) * penalty_local
Cost_edge1  = Q_norm * drift_edge1  + V(t) * penalty_edge1
Cost_edge2  = Q_norm * drift_edge2  + V(t) * penalty_edge2
Cost_edge3  = Q_norm * drift_edge3  + V(t) * penalty_edge3

Decision = argmin(Cost_local, Cost_edge1, Cost_edge2, Cost_edge3)
```

---

## Project Structure

```
FP/
|-- edge_server.py          # TCP edge server - executes real computation tasks
|-- lyapunov_engine.py      # RS-PLO algorithm engine + multi-edge DPP + static baseline
|-- demo.py                 # Multi-edge interactive CLI - process YOUR data in real-time
|-- dashboard.py            # Live web dashboard backend (SSE + HTTP server)
|-- dashboard.html          # Real-time Chart.js dashboard UI (dark theme)
|-- run_experiment.py       # Automated experiment - RS-PLO vs Static comparison
|-- test_engine.py          # Unit tests for engine logic
|-- Dockerfile              # Docker containerization
|-- docker-compose.yml      # Docker Compose for one-command setup
|-- README.md               # This file
|-- paper_text.txt          # Extracted paper text
+-- results/                # Generated comparison plots
    |-- 01_rsplo_mechanism.png
    |-- 02_comparison.png
    +-- 03_summary.png
```

### File Details

| File | Purpose |
|------|---------|
| `edge_server.py` | TCP server with 7 task handlers, threaded client connections |
| `lyapunov_engine.py` | RS-PLO + Static Lyapunov engines, multi-edge DPP, channel model |
| `demo.py` | Multi-edge interactive CLI with per-server history and colors |
| `dashboard.py` | SSE backend + HTTP server for real-time browser dashboard |
| `dashboard.html` | Chart.js animated charts, server cards, decision feed |
| `run_experiment.py` | Orchestrates full experiment, generates Matplotlib plots |
| `test_engine.py` | Unit tests validating engine, DPP decisions, multi-edge logic |

---

## Installation

### Prerequisites

- **Python 3.8+** (tested with 3.13.7)
- **numpy** -- numerical computation
- **matplotlib** -- plot generation (only for `run_experiment.py`)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/barathkrishnan111/RS-PLO-Lyapunov-Based-Dynamic-Task-Offloading-in-Edge-Computing.git
cd RS-PLO-Lyapunov-Based-Dynamic-Task-Offloading-in-Edge-Computing

# 2. Install dependencies
pip install numpy matplotlib
```

No other setup needed. Edge servers start automatically.

---

## Usage

### 1. Live Dashboard (`dashboard.py`)

**The best way to demonstrate the system.** Starts a web dashboard with real-time charts:

```bash
python dashboard.py
```

Then open **http://localhost:8080** in your browser.

**What you get:**
- **Live stats** -- Tasks processed, Q(t), Z(t), V(t), offload ratio
- **Server cards** -- LOCAL + 3 edge servers with task counts and percentages
- **Animated charts** -- Q(t)/Z(t) and V(t)/Energy in real-time
- **Decision feed** -- Color-coded scrolling log of every task decision
- **CLI** -- Same interactive commands as `demo.py` in the terminal

Type commands in the terminal while watching the dashboard update live:

```
rsplo> matrix 100          # Watch the chart update!
rsplo> tunnel              # See volatility spike in the chart
rsplo> burst               # Rapid-fire 5 tasks
rsplo> servers             # View all 3 edge server signal quality
```

### 2. Interactive CLI Demo (`demo.py`)

Terminal-only mode (no browser needed):

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
rsplo> tunnel              # Enter tunnel (very weak signal, Z spikes)
rsplo> distance 500        # Set exact distance in meters
```

#### System Commands

```
rsplo> status              # Show Q(t), Z(t), V(t), channel quality
rsplo> servers             # Show all 3 edge servers + signal quality
rsplo> burst               # Send 5 rapid tasks (stress test)
rsplo> history             # Show full task execution log with per-server breakdown
rsplo> help                # List all commands
rsplo> quit                # Exit
```

#### Example Session (Multi-Edge)

```
rsplo> matrix 100
  +--- Task #1: Matrix 100x100 ---
  |  Channel: -89.5dB | Distance: 300m | TX Rate: 27.2 Mbps
  |  Q=0 | Z=0.000 | V=10.0000
  |  Decision: >> OFFLOAD -> Edge-1 (Near)
  |  Done! Executed on Edge-1 (Near) in 18.13ms

rsplo> tunnel
  ENTERED TUNNEL -- Very weak signal! Distance: 2400m

rsplo> matrix 100
  +--- Task #2: Matrix 100x100 ---
  |  Channel: -118.9dB | Distance: 2408m | TX Rate: 0.1 Mbps
  |  Q=409,600 | Z=6.477 | V=0.0000
  |  Decision: >> LOCAL -> This Device
  |  Done! Executed on LOCAL in 2.02ms

rsplo> move close

rsplo> burst
  BURST MODE -- Sending 5 rapid tasks...
  Task #3-7 all offloaded to Edge-1 (Near)

rsplo> history
  Total: 7 tasks
    Edge-1 (Near): 6 (86%)
    LOCAL: 1 (14%)
```

Notice how the **same task type** gets different decisions based on channel conditions and server availability!

---

### 3. Automated Experiment (`run_experiment.py`)

Runs a full comparison between RS-PLO (adaptive) and Static Lyapunov (fixed V):

```bash
python run_experiment.py
```

**What it does:**
1. Starts 3 edge servers (ports 9999, 10000, 10001)
2. Runs RS-PLO with 5 users over 200 time slots
3. Runs Static baseline with the same conditions
4. Generates 3 comparison plots in `results/`
5. Prints summary statistics with per-server breakdown

**Output plots:**

| Plot | Shows |
|------|-------|
| `01_rsplo_mechanism.png` | Q(t), Z(t), V(t) over time -- the adaptive mechanism |
| `02_comparison.png` | RS-PLO vs Static: queue, energy, offload pattern, latency |
| `03_summary.png` | Per-user comparison and overall metrics |

---

### 4. Standalone Edge Server

Run the edge server independently:

```bash
# Terminal 1: Start a server on port 9999
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

All tasks are **real computation** -- actual matrix products, actual hashing, actual factorization.

---

## System Parameters

Configurable in `lyapunov_engine.py` (`SystemParams` class):

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `V_max` | 10.0 | Maximum control parameter (energy weight when stable) |
| `beta` | 2.0 | Sensitivity of V(t) to volatility (higher = faster response) |
| `gamma` | 0.2 | Volatility queue decay rate (lower = longer memory of risk) |

### Edge Server Configuration

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `edge_servers` | 3 servers | List of `EdgeServerConfig` objects |
| `distance` | 150/600/1200m | Distance to each edge server |
| `compute_multiplier` | 1.0/1.5/2.0x | Server compute capability |
| `port` | 9999/10000/10001 | TCP port for each server |

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

---

## How the Decision Works

For each incoming task, the algorithm computes costs for **all 4 options**:

```
Cost_local   = Q_norm x drift_local   + V(t) x penalty_local
Cost_edge1   = Q_norm x drift_edge1   + V(t) x penalty_edge1
Cost_edge2   = Q_norm x drift_edge2   + V(t) x penalty_edge2
Cost_edge3   = Q_norm x drift_edge3   + V(t) x penalty_edge3
```

**Pick the option with the lowest cost.**

The drift captures **queue pressure**, the penalty captures **energy cost**, and V(t) controls the balance:

- **V = 10** (stable): Energy matters a lot -- pick the cheapest option
- **V = 0** (volatile): Energy is irrelevant -- pick the fastest option
- **Large Q**: Queue pressure dominates -- drain the queue ASAP

---

## Technical Details

### Channel Model
- Path loss: `h(d) = h0 * d^(-alpha)` with Rayleigh fading
- Per-server channel: `h_server(d) = h0 * |d - d_server|^(-alpha)`
- Shannon capacity: `R = B * log2(1 + P*h/N0)`
- Prediction error computed in **dB scale** for meaningful volatility tracking

### Communication Protocol
- TCP sockets with length-prefixed JSON (4-byte big-endian header)
- Threaded server handles multiple concurrent clients
- Automatic fallback to local execution if edge connection fails

### Dashboard Architecture
- Python `http.server` serves `dashboard.html` on port 8080
- **Server-Sent Events (SSE)** push real-time data to the browser
- Zero external dependencies (no WebSockets library needed)
- Chart.js for animated, responsive charts

### Energy Model
- Local: `E_local = energy_per_bit * task_bits` (slow CPU = more energy per bit)
- Offload: `E_offload = P_tx * (task_bits / R)` (depends on channel to specific server)
- When channel is good: offload is cheaper. When bad: local is cheaper.

---

## Docker

Run the entire system with one command:

```bash
# Build and run
docker-compose up --build

# Open dashboard
open http://localhost:8080
```

Or manually:

```bash
docker build -t rsplo .
docker run -it -p 8080:8080 rsplo
```

---

## License

Academic/research use. Based on the RS-PLO framework from the referenced paper.