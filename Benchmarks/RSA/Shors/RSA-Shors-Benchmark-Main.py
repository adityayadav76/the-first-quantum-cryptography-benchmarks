# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Modification:
# Changes have been made to execute this program on Automatski' Quantum Computers
# instead of the Qiskit Aer Simulator

"""
Shor's factoring algorithm
"""

############## Loading Libraries ##############
import psutil
import array
import fractions
import logging
import math
import sys
import time
from typing import Optional, List, Union, Dict, Any
import statistics as stats
import numpy as np
import sys
sys.path.append('../../python/')
from AutomatskiKomencoQiskit import *

# remove Qiskit verbosity
for name in [
    "qiskit",
    "qiskit_aer",
    "qiskit.compiler",
    "qiskit.transpiler",
    "qiskit.passmanager",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate, Instruction, ParameterVector
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit.transpiler import CouplingMap

from qiskit_algorithms.utils.validation import validate_min
from qiskit_algorithms.algorithm_result import AlgorithmResult
from qiskit_algorithms.exceptions import AlgorithmError

############## Helper Functions for the Benchmark ##############
def _get_rss_mb() -> Optional[float]:
    try:
        import os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None

def _get_peak_rss_mb() -> Optional[float]:
    try:
        if sys.platform == "win32":
            p = psutil.Process()
            try:
                return p.memory_full_info().peak_wset / (1024 * 1024)
            except Exception:
                return p.memory_info().rss / (1024 * 1024)
        else:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                # macOS: bytes
                return ru / (1024 * 1024)
            else:
                # Linux: KiB
                return ru / 1024
    except Exception:
        return None

############## Shor's Main Algorithm ##############
class Shor:
    def __init__(
        self,
        simulator: Optional[AerSimulator] = None,
        shots: int = 20_000,
        transpile_to_line: bool = True,
        max_bond_dimension: Optional[int] = None,
        truncation_threshold: Optional[float] = None,
    ) -> None:
        self.backend = simulator

        def try_set(key, val):
            try:
                self.backend.set_options(**{key: val})
            except Exception:
                pass

        # try different Aer option names across versions
        if max_bond_dimension is not None:
            for k in [
                "matrix_product_state_max_bond_dimension",
                "mps_max_bond_dimension",
                "max_bond_dimension",
            ]:
                try_set(k, max_bond_dimension)

        if truncation_threshold is not None:
            for k in [
                "matrix_product_state_truncation_threshold",
                "mps_truncation_threshold",
                "truncation_threshold",
            ]:
                try_set(k, truncation_threshold)

        self._shots = shots
        self._transpile_to_line = transpile_to_line
        self.last_transpile_s: float = 0.0  # set each factor() call

    # ---------- utility gates ----------
    @staticmethod
    def _get_angles(a: int, n: int) -> np.ndarray:
        bits_le = (bin(int(a))[2:].zfill(n))[::-1]
        angles = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(i + 1):
                if bits_le[j] == "1":
                    angles[i] += pow(2.0, -(i - j))
        return angles * np.pi

    @staticmethod
    def _phi_add_gate(angles: Union[np.ndarray, ParameterVector]) -> Gate:
        qc = QuantumCircuit(len(angles), name="phi_add_a")
        for i, ang in enumerate(angles):
            qc.p(ang, i)
        return qc.to_gate()

    def _double_controlled_phi_add_mod_N(
        self,
        angles: Union[np.ndarray, ParameterVector],
        c_phi_add_N: Gate,
        iphi_add_N: Gate,
        qft: Gate,
        iqft: Gate,
    ) -> QuantumCircuit:
        ctrl_q = QuantumRegister(2, "ctrl")
        b_q = QuantumRegister(len(angles), "b")
        flag = QuantumRegister(1, "flag")
        qc = QuantumCircuit(ctrl_q, b_q, flag, name="ccphi_add_a_mod_N")

        cc_add = self._phi_add_gate(angles).control(2)
        cc_sub = cc_add.inverse()

        qc.append(cc_add, [*ctrl_q, *b_q])
        qc.append(iphi_add_N, b_q)

        qc.append(iqft, b_q)
        qc.cx(b_q[-1], flag[0])
        qc.append(qft, b_q)

        qc.append(c_phi_add_N, [*flag, *b_q])

        qc.append(cc_sub, [*ctrl_q, *b_q])

        qc.append(iqft, b_q)
        qc.x(b_q[-1])
        qc.cx(b_q[-1], flag[0])
        qc.x(b_q[-1])
        qc.append(qft, b_q)

        qc.append(cc_add, [*ctrl_q, *b_q])

        return qc

    def _controlled_multiple_mod_N(
        self, n: int, N: int, a: int, c_phi_add_N: Gate, iphi_add_N: Gate, qft: Gate, iqft: Gate
    ) -> Instruction:
        ctrl = QuantumRegister(1, "ctrl")
        x = QuantumRegister(n, "x")
        b = QuantumRegister(n + 1, "b")
        flag = QuantumRegister(1, "flag")
        qc = QuantumCircuit(ctrl, x, b, flag, name="cmult_a_mod_N")

        angle_params = ParameterVector("angles", length=n + 1)
        mod_adder = self._double_controlled_phi_add_mod_N(angle_params, c_phi_add_N, iphi_add_N, qft, iqft)

        def append_adder(adder: QuantumCircuit, const: int, idx: int):
            partial = (pow(2, idx, N) * const) % N
            angles = self._get_angles(partial, n + 1)
            bound = adder.assign_parameters({angle_params: angles})
            qc.append(bound, [*ctrl, x[idx], *b, *flag])

        qc.append(qft, b)
        for i in range(n):
            append_adder(mod_adder, a, i)
        qc.append(iqft, b)

        for i in range(n):
            qc.cswap(ctrl, x[i], b[i])

        qc.append(qft, b)
        a_inv = pow(a, -1, mod=N) if sys.version_info >= (3, 8) else self.modinv(a, N)
        mod_adder_inv = mod_adder.inverse()
        for i in reversed(range(n)):
            append_adder(mod_adder_inv, a_inv, i)
        qc.append(iqft, b)

        return qc.to_instruction()

    def _power_mod_N(self, n: int, N: int, a: int) -> Instruction:
        up = QuantumRegister(2 * n, "up")
        down = QuantumRegister(n, "down")
        aux = QuantumRegister(n + 2, "aux")
        qc = QuantumCircuit(up, down, aux, name=f"{a}^x mod {N}")

        qft = QFT(n + 1, do_swaps=False).to_gate()
        iqft = qft.inverse()

        phiN = self._phi_add_gate(self._get_angles(N, n + 1))
        iphiN = phiN.inverse()
        c_phiN = phiN.control(1)

        for i in range(2 * n):
            partial_a = pow(a, pow(2, i), N)
            mult = self._controlled_multiple_mod_N(n, N, partial_a, c_phiN, iphiN, qft, iqft)
            qc.append(mult, [up[i], *down, *aux])

        return qc.to_instruction()

    # ---------- validation & helpers ----------
    @staticmethod
    def _validate_input(N: int, a: int):
        validate_min("N", N, 3)
        validate_min("a", a, 2)
        if N < 1 or N % 2 == 0:
            raise ValueError("N must be an odd integer > 1.")
        if a >= N or math.gcd(a, N) != 1:
            raise ValueError("Require 1 < a < N and gcd(a, N) = 1.")

    @staticmethod
    def modinv(a: int, m: int) -> int:
        def egcd(x, y):
            if x == 0:
                return y, 0, 1
            g, u, v = egcd(y % x, x)
            return g, v - (y // x) * u, u

        g, x, _ = egcd(a, m)
        if g != 1:
            raise ValueError(f"No modular inverse for a={a}, m={m} (gcd={g}).")
        return x % m

    # ---------- circuit builder ----------
    def construct_circuit(self, N: int, a: int = 2, measurement: bool = False) -> QuantumCircuit:
        self._validate_input(N, a)
        n = N.bit_length()

        up = QuantumRegister(2 * n, "up")
        down = QuantumRegister(n, "down")
        aux = QuantumRegister(n + 2, "aux")
        qc = QuantumCircuit(up, down, aux, name=f"Shor(N={N}, a={a})")

        qc.h(up)
        qc.x(down[0])
        qc.append(self._power_mod_N(n, N, a), qc.qubits)
        iqft = QFT(len(up)).inverse().to_gate()
        qc.append(iqft, up)

        if measurement:
            cm = ClassicalRegister(2 * n, "m")
            qc.add_register(cm)
            qc.measure(up, cm)
        return qc

    # ---------- post-processing ----------
    def _get_factors(self, N: int, a: int, measurement) -> Optional[List[int]]:
        if isinstance(measurement, (tuple, list)):
            bitstr = "".join(str(b) for b in measurement)
        elif isinstance(measurement, int):
            bitstr = format(measurement, f"0{2 * N.bit_length()}b")
        elif isinstance(measurement, str):
            bitstr = measurement.replace(" ", "")
        else:
            raise AlgorithmError(f"Bad measurement type {type(measurement)}")

        x_final = int(bitstr, 2)
        if x_final <= 0:
            return None

        T = 2 ** len(bitstr)
        x_over_T = x_final / T

        i = 0
        b = array.array("i")
        t = array.array("f")
        b.append(math.floor(x_over_T))
        t.append(x_over_T - b[0])

        while i < N:
            if i > 0:
                b.append(math.floor(1 / t[i - 1]))
                t.append((1 / t[i - 1]) - b[i])  # type: ignore
            denom = self._calculate_continued_fraction(b)
            i += 1

            if denom % 2 == 1:
                continue

            exp = pow(a, denom // 2) if denom < 1000 else 1_000_001_000
            if exp > 1_000_000_000:
                return None

            f1 = math.gcd(int(exp + 1), N)
            f2 = math.gcd(int(exp - 1), N)
            if any(f in {1, N} for f in (f1, f2)):
                if t[i - 1] == 0:
                    return None
            else:
                return sorted((f1, f2))
        return None

    @staticmethod
    def _calculate_continued_fraction(b: array.array) -> int:
        x_over_T = 0.0
        for i in reversed(range(len(b) - 1)):
            x_over_T = 1.0 / (b[i + 1] + x_over_T)
        x_over_T += b[0]
        return fractions.Fraction(x_over_T).limit_denominator().denominator

 # ---------- transpile & run ----------
    def _transpile_safe_basis(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Uses gates accpeted by the MSP
        """
        line = CouplingMap.from_line(circuit.num_qubits) if self._transpile_to_line else None
        start = time.perf_counter()

        # Flatten library blocks (QFT, etc.) before the preset pass manager sees them.
        circ_in = circuit.decompose(reps=5)

        safe_basis = ['ccx', 'ccz', 'cp', 'crz', 'cs', 'csdg', 'cswap', 'cu', 'cx', 'cy', 'cz', 'h', 'id', 'measure', 'p', 'rx', 'ry', 'rz', 's', 'sdg', 'swap', 'sx', 'sxdg', 't', 'tdg', 'u', 'x', 'y', 'z']

        try:
            circ_t = transpile(
                circ_in,
                coupling_map=line,
                layout_method="trivial" if line else None,
                basis_gates=safe_basis,
                optimization_level=3,
            )
        except Exception:
            circ_t = transpile(
                circ_in,
                basis_gates=safe_basis,
                optimization_level=3,
            )

        #final light decompose in case any composite snuck through.
        #circ_t = circ_t.decompose(reps=10)

        self.last_transpile_s = (time.perf_counter() - start)
        return circ_t


    def factor(self, N: int, a: int = 2, return_benchmark: bool = False):
        """Execute using AerSimulator(MPS), and returns benchmarking info."""
        bench: Dict[str, Any] = {
            "transpile_s": None,
            "run_s": None,
            "cpu_pct": None,
            "rss_mb_after": None,   # absolute RSS after run
            "rss_delta_mb": None,
            "peak_rss_mb": None,
        }

        rss_before = _get_rss_mb()

        result = ShorResult()

        # Construct + transpile
        circ = self.construct_circuit(N, a, measurement=True)
        circ = self._transpile_safe_basis(circ)
        bench["transpile_s"] = self.last_transpile_s

        # Run & time (seconds), measure CPU time too.
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        
        r = self.backend.run(circ, repetitions=self._shots,topK=1000)
        #r = job.result()
        
        run_wall = time.perf_counter() - t0_wall
        run_cpu = time.process_time() - t0_cpu
        bench["run_s"] = run_wall
        bench["cpu_pct"] = (100.0 * run_cpu / run_wall) if run_wall > 0 else None  # may exceed 100% on multi-core

        # Counts
        try:
            counts = r.get_counts(None)
            #print(counts)
        except Exception:
            counts = r.get_counts(0)
        if not counts:
            raise AlgorithmError("Simulation returned no counts (did it fail?).")

        cleaned_counts = {k.replace(" ", ""): v for k, v in counts.items()}
        result.total_counts = sum(cleaned_counts.values())
        for bitstr, _c in cleaned_counts.items():
            facs = self._get_factors(N, a, bitstr)
            if facs and facs not in result.factors:
                result.factors.append(facs)
                result.successful_counts += 1

        # RAM metrics
        rss_after = _get_rss_mb()
        bench["rss_mb_after"] = rss_after
        if rss_before is not None and rss_after is not None:
            bench["rss_delta_mb"] = rss_after - rss_before
        bench["peak_rss_mb"] = _get_peak_rss_mb()

        return (result, bench) if return_benchmark else result


class ShorResult(AlgorithmResult):
    """Result container for Shor."""
    def __init__(self) -> None:
        super().__init__()
        self._factors: List[List[int]] = []
        self._total_counts: int = 0
        self._successful_counts: int = 0

    @property
    def factors(self) -> List[List[int]]:
        return self._factors
    @factors.setter
    def factors(self, value: List[List[int]]) -> None:
        self._factors = value

    @property
    def total_counts(self) -> int:
        return self._total_counts
    @total_counts.setter
    def total_counts(self, value: int) -> None:
        self._total_counts = value

    @property
    def successful_counts(self) -> int:
        return self._successful_counts
    @successful_counts.setter
    def successful_counts(self, value: int) -> None:
        self._successful_counts = value


############## Averaging benchmark runner (with std) ##############
def run_benchmark_avg(shor: Shor, N: int, a: int, runs: int = 5) -> Dict[str, Optional[float]]:
    """
    Run shor.factor(..., return_benchmark=True) `runs` times and return mean and std for:
      - transpile_s
      - run_s
      - avg_cores (from cpu_pct)
      - machine_util_pct (normalized by logical cores)
      - rss_mb_after   (absolute RAM after each run)
      - peak_rss_mb
    """
    cores = psutil.cpu_count(logical=True) or 1

    vals: Dict[str, list] = {
        "transpile_s": [],
        "run_s": [],
        "avg_cores": [],
        "machine_util_pct": [],
        "rss_mb_after": [],
        "peak_rss_mb": [],
    }

    last_factors: List[List[int]] = []

    for _ in range(runs):
        res, bench = shor.factor(N, a, return_benchmark=True)
        if res.factors:
            last_factors = res.factors

        ts = bench.get("transpile_s")
        rs = bench.get("run_s")
        cp = bench.get("cpu_pct")
        rm = bench.get("rss_mb_after")
        pr = bench.get("peak_rss_mb")

        avg_cores = (cp / 100.0) if cp is not None else None
        machine_util_pct = (100.0 * avg_cores / cores) if avg_cores is not None else None

        if ts is not None:
            vals["transpile_s"].append(ts)
        if rs is not None:
            vals["run_s"].append(rs)
        if avg_cores is not None:
            vals["avg_cores"].append(avg_cores)
        if machine_util_pct is not None:
            vals["machine_util_pct"].append(machine_util_pct)
        if rm is not None:
            vals["rss_mb_after"].append(rm)
        if pr is not None:
            vals["peak_rss_mb"].append(pr)

    def mean_std(key: str):
        data = vals[key]
        if not data:
            return None, None
        if len(data) == 1:
            return float(data[0]), 0.0
        return float(sum(data) / len(data)), float(stats.stdev(data))

    results: Dict[str, Optional[float]] = {"runs": float(runs)}
    for k in vals.keys():
        m, s = mean_std(k)
        results[f"{k}_mean"] = m
        results[f"{k}_std"] = s

    results["_factors_example"] = last_factors
    return results


# ---------------- example usage (local, no IBM) ----------------
if __name__ == "__main__":
    qc = AutomatskiKomencoQiskit(host="xxx.xxx.xxx.xxx", port=xxx)
    shor = Shor(qc, shots=100000, transpile_to_line=True)
    
    N, a = 15, 4
    #N, a = 21, 8
    #N, a = 143, 21 #12
    #N, a = 1363, 46 #610 # p = 47 ; q = 29
    #N, a = 67297, 3113 #503 #50 #p = 389, q = 173
    #N, a = 1398488603, 737312150 #p = 37049, q = 37747 #10 digits
    #N, a = 85143280699972919909,737312150 #p = 9625297147 ; q = 8845782047   
    
    circuit = shor.construct_circuit(N, a, measurement=True)    
    # --- quick gate/depth summary ---
    def summarize(label, qc):
        ops = dict(qc.count_ops())
        # ignore no-ops if present
        for k in ("barrier", "id"):
            ops.pop(k, None)
        twoq = sum(ops.get(g, 0) for g in ('ccx', 'ccz', 'cp', 'crz', 'cs', 'csdg', 'cswap', 'cu', 'cx', 'cy', 'cz', 'swap'))
        print(f"{label}")
        #Depth: the longest path throught the circuit in terms of sequential gates
        print(f"Depth                 = {qc.depth()}")  #  |  size = {qc.size()}  |  2-qubit ops = {twoq}")
        #Size: the total number of gate in the circuit
        print(f"Size                  = {qc.size()}")
        #2-qubit ops: total operation of 2 qubit operations; important for error and runtime
        print(f"2-qubit ops           = {twoq}")
        print(f"Gates' Counts         = {ops}")

    #visualizing the circuit
    #circuit.draw("mpl")
    
    #couting gates after transpilation
    t_circuit = shor._transpile_safe_basis(circuit)
    summarize("\n=== Quantum Circuit After Transpilation ===", t_circuit)
    
    #printing benchmark results    
    print("\n=== Benchmark Summary for Factoring Using QFT ===")
    
    print(f"\nTotal qubits: {circuit.num_qubits}")
    
    RUNS = 1

    stats_avgs = run_benchmark_avg(shor, N, a, runs=RUNS)

    #results
    fx = stats_avgs.get("_factors_example") or []
    #print("\n=== Results for the Factoring Using QFT ===")
    print(f"N = {N}")
    print(f"a = {a}")
    if not fx:
        print("Factors not found.")
    else:
        #print("Factors:")
        for i, pair in enumerate(fx, 1):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                p, q = pair
                print(f"The factors of {N} are {p} and {q}.")
            else:
                print(f"{i}. {pair}")


    def fmt(x, nd=4):
        return "n/a" if x is None else f"{x:.{nd}f}"

    print(f"Runs:                       {int(stats_avgs['runs'])}")
    print(f"Avg transpilation (s):      {fmt(stats_avgs['transpile_s_mean'])} ± {fmt(stats_avgs['transpile_s_std'])}")
    print(f"Avg run time (s):           {fmt(stats_avgs['run_s_mean'])} ± {fmt(stats_avgs['run_s_std'])}")
    print(f"Avg cores used:             {fmt(stats_avgs['avg_cores_mean'], 2)} ± {fmt(stats_avgs['avg_cores_std'], 2)}")
    print(f"Avg CPU usage (%):          {fmt(stats_avgs['machine_util_pct_mean'], 2)} ± {fmt(stats_avgs['machine_util_pct_std'], 2)}")
    print(f"Avg RAM (MiB):              {fmt(stats_avgs['rss_mb_after_mean'], 2)} ± {fmt(stats_avgs['rss_mb_after_std'], 2)}")
    print(f"Avg Peak RAM (MiB):         {fmt(stats_avgs['peak_rss_mb_mean'], 2)} ± {fmt(stats_avgs['peak_rss_mb_std'], 2)}")


