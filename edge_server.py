"""
Edge Server — TCP server that receives and executes real computation tasks.
Part of the RS-PLO (Risk-Sensitive Predictive Lyapunov Optimization) system.

Runs as a separate process, simulating an MEC (Mobile Edge Computing) server.
Accepts tasks over TCP, executes them, and returns results + timing.
"""

import socket
import threading
import json
import time
import hashlib
import numpy as np
import struct
import sys


# ─────────────────────────────────────────────
#  REAL COMPUTATION TASKS
# ─────────────────────────────────────────────

def task_matrix_multiply(size: int) -> dict:
    """Real matrix multiplication — actual CPU work."""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    start = time.perf_counter()
    C = np.dot(A, B)
    elapsed = time.perf_counter() - start
    return {
        "result_checksum": float(np.sum(C)),
        "exec_time": elapsed,
        "task_type": "matrix_multiply",
        "size": size
    }


def task_hash_data(size_bytes: int) -> dict:
    """SHA-256 hashing of random data — real crypto work."""
    data = np.random.bytes(size_bytes)
    start = time.perf_counter()
    h = hashlib.sha256(data)
    for _ in range(100):  # multiple rounds to make it measurable
        h = hashlib.sha256(h.digest() + data[:1024])
    elapsed = time.perf_counter() - start
    return {
        "result_hash": h.hexdigest()[:16],
        "exec_time": elapsed,
        "task_type": "hash_data",
        "size": size_bytes
    }


def task_prime_factorize(n: int) -> dict:
    """Trial-division prime factorization — real CPU work."""
    start = time.perf_counter()
    factors = []
    original = n
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    elapsed = time.perf_counter() - start
    return {
        "factors_count": len(factors),
        "largest_factor": factors[-1] if factors else original,
        "exec_time": elapsed,
        "task_type": "prime_factorize",
        "n": original
    }


def task_word_count(text: str) -> dict:
    """Count words, lines, and characters in text — real text processing."""
    start = time.perf_counter()
    words = text.split()
    lines = text.count('\n') + 1
    chars = len(text)
    # Frequency analysis
    word_freq = {}
    for w in words:
        w_lower = w.lower().strip('.,!?;:')
        word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:10]
    elapsed = time.perf_counter() - start
    return {
        "word_count": len(words),
        "line_count": lines,
        "char_count": chars,
        "unique_words": len(word_freq),
        "top_10_words": top_words,
        "exec_time": elapsed,
        "task_type": "word_count"
    }


def task_sort_numbers(numbers: list) -> dict:
    """Sort a list of numbers — real computation."""
    start = time.perf_counter()
    sorted_nums = sorted(numbers)
    elapsed = time.perf_counter() - start
    return {
        "count": len(sorted_nums),
        "min": sorted_nums[0] if sorted_nums else None,
        "max": sorted_nums[-1] if sorted_nums else None,
        "median": sorted_nums[len(sorted_nums)//2] if sorted_nums else None,
        "mean": sum(sorted_nums) / len(sorted_nums) if sorted_nums else None,
        "exec_time": elapsed,
        "task_type": "sort_numbers"
    }


def task_text_encrypt(text: str, shift: int = 13) -> dict:
    """Caesar cipher encryption — real crypto processing."""
    start = time.perf_counter()
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    encrypted = ''.join(result)
    # Also compute SHA-256 hash
    h = hashlib.sha256(text.encode()).hexdigest()
    elapsed = time.perf_counter() - start
    return {
        "encrypted_preview": encrypted[:200],
        "original_length": len(text),
        "sha256": h[:32],
        "exec_time": elapsed,
        "task_type": "text_encrypt"
    }


def task_file_stats(data: str) -> dict:
    """Analyze file content — frequency, entropy, patterns."""
    start = time.perf_counter()
    import math
    # Character frequency
    freq = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(data)
    # Shannon entropy
    entropy = 0
    for count in freq.values():
        p = count / total if total > 0 else 0
        if p > 0:
            entropy -= p * math.log2(p)
    # Find patterns
    sentences = data.count('.') + data.count('!') + data.count('?')
    paragraphs = data.count('\n\n') + 1
    elapsed = time.perf_counter() - start
    return {
        "size_bytes": len(data.encode()),
        "unique_chars": len(freq),
        "entropy_bits": round(entropy, 4),
        "sentences": sentences,
        "paragraphs": paragraphs,
        "exec_time": elapsed,
        "task_type": "file_stats"
    }


TASK_HANDLERS = {
    "matrix_multiply": lambda params: task_matrix_multiply(params.get("size", 50)),
    "hash_data": lambda params: task_hash_data(params.get("size_bytes", 50000)),
    "prime_factorize": lambda params: task_prime_factorize(params.get("n", 999983)),
    "word_count": lambda params: task_word_count(params.get("text", "")),
    "sort_numbers": lambda params: task_sort_numbers(params.get("numbers", [])),
    "text_encrypt": lambda params: task_text_encrypt(params.get("text", ""), params.get("shift", 13)),
    "file_stats": lambda params: task_file_stats(params.get("data", "")),
}


# ─────────────────────────────────────────────
#  TCP MESSAGE PROTOCOL
# ─────────────────────────────────────────────

def send_message(sock: socket.socket, data: dict):
    """Send a length-prefixed JSON message."""
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_message(sock: socket.socket) -> dict:
    """Receive a length-prefixed JSON message."""
    raw_len = _recv_exactly(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack("!I", raw_len)[0]
    raw_data = _recv_exactly(sock, msg_len)
    if not raw_data:
        return None
    return json.loads(raw_data.decode("utf-8"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ─────────────────────────────────────────────
#  CLIENT HANDLER
# ─────────────────────────────────────────────

def handle_client(conn: socket.socket, addr, server_stats: dict):
    """Handle a single client connection."""
    try:
        while True:
            msg = recv_message(conn)
            if msg is None:
                break

            task_type = msg.get("task_type", "matrix_multiply")
            params = msg.get("params", {})
            task_id = msg.get("task_id", -1)

            handler = TASK_HANDLERS.get(task_type)
            if handler is None:
                send_message(conn, {"error": f"Unknown task type: {task_type}"})
                continue

            # Execute the real task
            receive_time = time.perf_counter()
            result = handler(params)
            result["task_id"] = task_id
            result["server_receive_time"] = receive_time
            result["status"] = "completed"

            server_stats["tasks_completed"] += 1
            send_message(conn, result)

    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
        pass
    finally:
        conn.close()


# ─────────────────────────────────────────────
#  EDGE SERVER
# ─────────────────────────────────────────────

class EdgeServer:
    """
    MEC Edge Server — accepts computation tasks over TCP and executes them.
    Runs in its own process to simulate a real edge computing node.
    """

    def __init__(self, host="127.0.0.1", port=9999):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.stats = {"tasks_completed": 0}

    def start(self):
        """Start the edge server (blocking)."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        self.server_socket.settimeout(1.0)
        self.running = True

        print(f"[EDGE SERVER] Listening on {self.host}:{self.port}")
        print(f"[EDGE SERVER] Ready to accept computation tasks...")

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                t = threading.Thread(target=handle_client, args=(conn, addr, self.stats), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

        print(f"[EDGE SERVER] Shut down. Total tasks completed: {self.stats['tasks_completed']}")

    def stop(self):
        """Stop the edge server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    server = EdgeServer(port=port)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
