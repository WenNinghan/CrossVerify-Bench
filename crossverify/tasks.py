"""Verifiable 180-task multi-domain benchmark with deterministic evaluators & micro-checks.

Domains (60 tasks each = 180 tasks total; 20 Easy / 20 Medium / 20 Hard per domain):
    1. ``code_patch``: Code Patch & Invariant Repair (executable Python AST + unit assertions)
    2. ``scientific_symbolic``: Scientific Symbolic & Dimensional Reasoning (kinetics units,
       ODE Jacobian spectra, Turing dispersion stability)
    3. ``evidence_crosscheck``: Multi-Constraint Evidence Cross-Checking (multi-source
       conflicting table provenance + constrained task scheduling)

Design Principle (Zero Oracle Leakage at Inference Time)
--------------------------------------------------------
Each ``BenchmarkTask`` separates:
    - ``evaluate(candidate)``: Exact ground-truth oracle evaluator used **only** for final
      benchmark scoring (never exposed to any agent or router during execution).
    - ``runtime_microcheck(candidate)``: Cheap, zero-oracle runtime tool check (Python AST
      parse + public signature/type smoke test, dimensional conservation check, JSON schema
      & public mutex check) legitimately available to ``Tool-Augmented Verifiers`` and
      ``ParetoAdaptiveRouter``. Subtle semantic distractors pass ``runtime_microcheck``
      while failing ``evaluate``, ensuring tool checks complement rather than replace
      cross-model reasoning.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PromptFeatures:
    """Observable structural complexity signals extracted solely from prompt text."""

    char_length: int
    line_count: int
    constraint_clause_count: int
    symbolic_density: float
    code_or_table_markers: int
    complexity_score: float


@lru_cache(maxsize=512)
def extract_prompt_features(prompt: str) -> PromptFeatures:
    """Compute zero-oracle structural complexity score in [0, 1] from raw prompt text."""
    char_len = len(prompt)
    lines = prompt.strip().splitlines()
    line_cnt = len(lines)

    constraint_keywords = re.findall(
        r"\b(?:invariant|must|constraint|precedence|mutex|tie-break|authority|"
        r"eigenvalue|equilibrium|checksum|wavenumber|modulo|monotonic|strictly|"
        r"stale|override|boundary|conservation|capacity)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    clause_cnt = len(constraint_keywords)

    sym_chars = sum(1 for ch in prompt if ch in "[]{}()+-*/=<>^_|")
    sym_density = sym_chars / max(char_len, 1)

    markers = len(re.findall(r"(?:```|\|\s*\w+|\bJ\s*=|\bu\^\*|\bSource_\w+|\bRule\s+\d+)", prompt))

    # Calibrated smooth sigmoid combination purely over observable prompt signals
    raw = (
        0.34 * min(char_len / 1150.0, 1.25)
        + 0.30 * min(clause_cnt / 10.0, 1.25)
        + 0.20 * min(line_cnt / 22.0, 1.25)
        + 0.16 * min(markers / 6.0, 1.25)
    )
    score = float(np.clip(raw / 1.08, 0.0, 1.0))
    return PromptFeatures(
        char_length=char_len,
        line_count=line_cnt,
        constraint_clause_count=clause_cnt,
        symbolic_density=float(sym_density),
        code_or_table_markers=markers,
        complexity_score=score,
    )


@dataclass(frozen=True)
class BenchmarkTask:
    """Single verifiable problem instance in CrossVerify-Bench."""

    task_id: str
    domain: str
    difficulty: str
    prompt: str
    ground_truth: str
    distractors: Tuple[str, str, str]
    _oracle_fn: Callable[[str], bool]
    _microcheck_fn: Callable[[str], Tuple[bool, str]]
    _eval_cache: Dict[str, bool] = None  # type: ignore[assignment]
    _micro_cache: Dict[str, Tuple[bool, str]] = None  # type: ignore[assignment]
    _canon_cache: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_eval_cache", {})
        object.__setattr__(self, "_micro_cache", {})
        object.__setattr__(self, "_canon_cache", {})

    def evaluate(self, candidate: str) -> bool:
        """Exact deterministic ground-truth evaluation (benchmark scoring only)."""
        cached = self._eval_cache.get(candidate)
        if cached is not None:
            return cached
        try:
            res = bool(self._oracle_fn(candidate))
        except Exception:
            res = False
        if len(self._eval_cache) < 128:
            self._eval_cache[candidate] = res
        return res

    def runtime_microcheck(self, candidate: str) -> Tuple[bool, str]:
        """Lightweight zero-oracle runtime tool check (syntax, schema, public invariants)."""
        cached = self._micro_cache.get(candidate)
        if cached is not None:
            return cached
        try:
            res = self._microcheck_fn(candidate)
        except Exception as exc:
            res = (False, f"microcheck_exception:{type(exc).__name__}")
        if len(self._micro_cache) < 128:
            self._micro_cache[candidate] = res
        return res

    def canonicalize(self, candidate: str) -> str:
        """Normalize candidate string for self-consistency voting & discrepancy detection."""
        cached = self._canon_cache.get(candidate)
        if cached is not None:
            return cached
        text = candidate.strip()
        if self.domain in ("scientific_symbolic", "evidence_crosscheck"):
            try:
                obj = json.loads(text)
                res = json.dumps(obj, sort_keys=True, separators=(",", ":"))
            except Exception:
                res = re.sub(r"\s+", " ", text)
        else:
            lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
            res = "\n".join(lines)
        if len(self._canon_cache) < 128:
            self._canon_cache[candidate] = res
        return res


# ---------------------------------------------------------------------------
# Helper: Safe isolated Python snippet execution for Domain 1 (code_patch)
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "ValueError": ValueError,
}


def _compile_solve_fn(code_str: str, expected_arity: int) -> Tuple[Any, str]:
    """Parse AST and compile ``def solve(...)`` inside a restricted builtins namespace."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        return None, f"syntax_error:{exc.msg}"

    solve_defs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"
    ]
    if not solve_defs:
        return None, "missing_solve_function"
    fn_node = solve_defs[0]
    if len(fn_node.args.args) != expected_arity:
        return None, f"arity_mismatch:expected_{expected_arity}_got_{len(fn_node.args.args)}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None, "forbidden_import"

    env: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    exec(compile(tree, filename="<candidate_patch>", mode="exec"), env, env)
    return env.get("solve"), "ok"


# ---------------------------------------------------------------------------
# Domain 1 Builder: Code Patch & Invariant Repair (60 tasks)
# ---------------------------------------------------------------------------


def _build_code_patch_tasks() -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []

    # --- EASY (20 tasks): Clamped modular step-counting with inclusive bounds ---
    for i in range(1, 21):
        step = (i % 5) + 2
        offset = (i % 4) - 1
        cap = 40 + 5 * i
        tid = f"code_easy_{i:02d}"
        prompt = (
            f"[Task {tid}] Repair `def solve(lo, hi)` so it returns the count of integers x "
            f"in the closed interval [lo, hi] satisfying `(x - {offset}) % {step} == 0`, "
            f"clamped above by {cap}. Must return 0 when `lo > hi`.\n"
            f"```python\ndef solve(lo, hi):\n    # Buggy: off-by-one & ignores lo > hi\n"
            f"    return min({cap}, (hi - lo) // {step})\n```"
        )
        gt = (
            f"def solve(lo, hi):\n"
            f"    if lo > hi:\n"
            f"        return 0\n"
            f"    first = lo + (({offset} - lo) % {step})\n"
            f"    if first > hi:\n"
            f"        return 0\n"
            f"    cnt = ((hi - first) // {step}) + 1\n"
            f"    return min({cap}, cnt)\n"
        )
        # d0: syntax/arity broken; d1: returns negative when lo > hi (fails non-negative smoke check);
        # d2: subtle off-by-one missing cap or exact boundary alignment when first == hi
        d0 = f"def solve(lo):\n    return max(0, (hi - lo) // {step})\n"
        d1 = f"def solve(lo, hi):\n    return (hi - lo) // {step}\n"
        d2 = (
            f"def solve(lo, hi):\n"
            f"    if lo > hi:\n"
            f"        return 0\n"
            f"    first = lo + (({offset} - lo) % {step})\n"
            f"    if first >= hi:\n"
            f"        return 0\n"
            f"    return min({cap}, ((hi - first) // {step}) + 1)\n"
        )

        def make_code_easy_eval(st: int, off: int, cp: int) -> Tuple[Callable, Callable]:
            test_pairs = [
                (-10, 25),
                (5, -3),
                (off, off),
                (off, off + st * (cp + 4)),
                (-17, -17),
                (0, 100),
            ]

            def oracle(cand: str) -> bool:
                fn, status = _compile_solve_fn(cand, 2)
                if fn is None or status != "ok":
                    return False
                for a, b in test_pairs:
                    expected = (
                        0
                        if a > b
                        else min(cp, sum(1 for x in range(a, b + 1) if (x - off) % st == 0))
                    )
                    out = fn(a, b)
                    if not isinstance(out, int) or isinstance(out, bool) or out != expected:
                        return False
                return True

            def micro(cand: str) -> Tuple[bool, str]:
                fn, status = _compile_solve_fn(cand, 2)
                if fn is None:
                    return False, status
                # Public invariant smoke check: empty range [10, 2] must be >= 0 and int
                r_empty = fn(10, 2)
                r_basic = fn(0, 20)
                if not isinstance(r_empty, int) or r_empty < 0:
                    return False, "invariant_violation:negative_on_inverted_range"
                if not isinstance(r_basic, int) or r_basic < 0:
                    return False, "invariant_violation:invalid_return_type"
                return True, "ok"

            return oracle, micro

        orc, mic = make_code_easy_eval(step, offset, cap)
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="code_patch",
                difficulty="easy",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    # --- MEDIUM (20 tasks): Merged non-overlapping interval length with gap-bridge threshold ---
    for i in range(1, 21):
        bridge = i % 4  # merge intervals if next_start <= curr_end + bridge
        scale = (i % 3) + 1
        tid = f"code_med_{i:02d}"
        prompt = (
            f"[Task {tid}] Repair `def solve(intervals)` under two invariants:\n"
            f"  1. Filter out degenerate intervals where `start >= end`.\n"
            f"  2. Sort remaining intervals and coalesce adjacent intervals whenever "
            f"`next_start <= curr_end + {bridge}` (gap-bridge tolerance = {bridge}).\n"
            f"  3. Return `{scale} * sum(end - start)` across the final coalesced intervals.\n"
            f"Ensure monotonic boundary handling on duplicate and nested intervals.\n"
            f"```python\ndef solve(intervals):\n    intervals.sort()\n"
            f"    return sum(b - a for a, b in intervals)\n```"
        )
        gt = (
            f"def solve(intervals):\n"
            f"    valid = sorted((a, b) for a, b in intervals if a < b)\n"
            f"    if not valid:\n"
            f"        return 0\n"
            f"    merged = [list(valid[0])]\n"
            f"    for a, b in valid[1:]:\n"
            f"        if a <= merged[-1][1] + {bridge}:\n"
            f"            if b > merged[-1][1]:\n"
            f"                merged[-1][1] = b\n"
            f"        else:\n"
            f"            merged.append([a, b])\n"
            f"    return {scale} * sum(b - a for a, b in merged)\n"
        )
        # d0: syntax error; d1: fails degenerate negative interval smoke invariant;
        # d2: subtle nested-interval overwrite bug `merged[-1][1] = b` without `max`!
        d0 = f"def solve(intervals):\n    for a, b in sorted(intervals)\n        return 0\n"
        d1 = (
            f"def solve(intervals):\n"
            f"    return {scale} * sum(b - a for a, b in intervals)\n"
        )
        d2 = (
            f"def solve(intervals):\n"
            f"    valid = sorted((a, b) for a, b in intervals if a < b)\n"
            f"    if not valid:\n"
            f"        return 0\n"
            f"    merged = [list(valid[0])]\n"
            f"    for a, b in valid[1:]:\n"
            f"        if a <= merged[-1][1] + {bridge}:\n"
            f"            merged[-1][1] = b\n"
            f"        else:\n"
            f"            merged.append([a, b])\n"
            f"    return {scale} * sum(b - a for a, b in merged)\n"
        )

        def make_code_med_eval(br: int, sc: int) -> Tuple[Callable, Callable]:
            suites = [
                [],
                [(5, 2), (3, 3)],
                [(1, 10), (2, 5)],  # nested! catches d2!
                [(0, 3), (3 + br, 7), (8 + br, 10)],
                [(-5, -1), (-3, 4), (2, 3), (10, 12)],
            ]

            def ref_fn(ivs: List[Tuple[int, int]]) -> int:
                v = sorted((a, b) for a, b in ivs if a < b)
                if not v:
                    return 0
                m = [[v[0][0], v[0][1]]]
                for a, b in v[1:]:
                    if a <= m[-1][1] + br:
                        m[-1][1] = max(m[-1][1], b)
                    else:
                        m.append([a, b])
                return sc * sum(b - a for a, b in m)

            def oracle(cand: str) -> bool:
                fn, status = _compile_solve_fn(cand, 1)
                if fn is None or status != "ok":
                    return False
                for case in suites:
                    out = fn(list(case))
                    if not isinstance(out, int) or out != ref_fn(case):
                        return False
                return True

            def micro(cand: str) -> Tuple[bool, str]:
                fn, status = _compile_solve_fn(cand, 1)
                if fn is None:
                    return False, status
                # Smoke check: degenerate intervals [(6, 2), (4, 4)] must yield 0
                r_deg = fn([(6, 2), (4, 4)])
                if not isinstance(r_deg, int) or r_deg != 0:
                    return False, "invariant_violation:degenerate_interval_not_zero"
                return True, "ok"

            return oracle, micro

        orc, mic = make_code_med_eval(bridge, scale)
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="code_patch",
                difficulty="medium",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    # --- HARD (20 tasks): Constrained DAG Longest Weighted Path with Node-Quota & Modulo Bonus ---
    for i in range(1, 21):
        max_hops = (i % 3) + 3
        penalty = (i % 5) + 2
        cap_nodes = 5 + (i % 3)
        tid = f"code_hard_{i:02d}"
        prompt = (
            f"[Task {tid}] Repair `def solve(n, edges)` for a directed graph on nodes `0..n-1` "
            f"where each edge is `(u, v, w)`.\n"
            f"Specification & Invariants:\n"
            f"  - Rule 1 (Strict DAG Forward Edge Filter): Only edges strictly satisfying "
            f"`0 <= u < v < n` are valid transitions; self-loops and backward edges must be ignored.\n"
            f"  - Rule 2 (Hop Capacity Constraint): Path starts at node `0` and may take at most "
            f"`{max_hops}` edges (hops <= {max_hops}).\n"
            f"  - Rule 3 (Congestion Penalty): Every edge whose destination `v >= {cap_nodes}` "
            f"incurs an integer congestion deduction of `{penalty}` subtracted from `w`.\n"
            f"  - Rule 4 (Boundary Invariant): Return the maximum reachable path score starting at "
            f"node `0` (a 0-hop stay at node `0` has score `0`, so result must be `>= 0`; if `n <= 0`, return `0`).\n"
            f"  - Tie-break / Dominance: Track exact `(node, hops)` DP state so longer high-weight "
            f"paths are not shadowed by high-hop low-weight arrivals.\n"
            f"```python\ndef solve(n, edges):\n    dp = [0] * n\n    for u, v, w in edges:\n"
            f"        dp[v] = max(dp[v], dp[u] + w)\n    return max(dp)\n```"
        )
        gt = (
            f"def solve(n, edges):\n"
            f"    if n <= 0:\n"
            f"        return 0\n"
            f"    neg_inf = -10**15\n"
            f"    dp = [[neg_inf] * ({max_hops} + 1) for _ in range(n)]\n"
            f"    dp[0][0] = 0\n"
            f"    valid = sorted((u, v, w) for u, v, w in edges if 0 <= u < v < n)\n"
            f"    for u, v, w in valid:\n"
            f"        eff_w = w - ({penalty} if v >= {cap_nodes} else 0)\n"
            f"        for h in range({max_hops}):\n"
            f"            if dp[u][h] > neg_inf:\n"
            f"                cand = dp[u][h] + eff_w\n"
            f"                if cand > dp[v][h + 1]:\n"
            f"                    dp[v][h + 1] = cand\n"
            f"    best = 0\n"
            f"    for node in range(n):\n"
            f"        for h in range({max_hops} + 1):\n"
            f"            if dp[node][h] > best:\n"
            f"                best = dp[node][h]\n"
            f"    return best\n"
        )
        # d0: arity mismatch; d1: crashes or returns < 0 on n=0 / negative weights;
        # d2: passes smoke invariants (handles n<=0, backward edges, >=0), but collapses hop dimension
        # greedily or forgets sorting edges by u so out-of-order edge list misses multi-hop chain!
        d0 = f"def solve(edges):\n    return 0\n"
        d1 = (
            f"def solve(n, edges):\n"
            f"    return sum(w for u, v, w in edges)\n"
        )
        d2 = (
            f"def solve(n, edges):\n"
            f"    if n <= 0:\n"
            f"        return 0\n"
            f"    neg_inf = -10**15\n"
            f"    dp = [[neg_inf] * ({max_hops} + 1) for _ in range(n)]\n"
            f"    dp[0][0] = 0\n"
            f"    for u, v, w in edges:\n"
            f"        if 0 <= u < v < n:\n"
            f"            eff_w = w - ({penalty} if v > {cap_nodes} else 0)\n"
            f"            for h in range({max_hops}):\n"
            f"                if dp[u][h] > neg_inf and dp[u][h] + eff_w > dp[v][h + 1]:\n"
            f"                    dp[v][h + 1] = dp[u][h] + eff_w\n"
            f"    return max(0, max(max(row) for row in dp))\n"
        )

        def make_code_hard_eval(mh: int, pen: int, cn: int) -> Tuple[Callable, Callable]:
            # Unordered edges + exact v == cn boundary tests!
            suites = [
                (0, [(0, 1, 10)]),
                (4, [(2, 1, 50), (0, 1, -5), (1, 1, 100)]),
                (cn + 2, [(1, cn, 12), (0, 1, 10), (cn, cn + 1, 9)]),  # unordered & v == cn!
                (
                    cn + 3,
                    [
                        (0, 1, 5),
                        (1, 2, 5),
                        (2, 3, 5),
                        (3, 4, 5),
                        (4, 5, 5),
                        (5, 6, 50),
                        (0, cn, 18),
                    ],
                ),
            ]

            def ref_solve(n_val: int, edge_list: List[Tuple[int, int, int]]) -> int:
                if n_val <= 0:
                    return 0
                neg_inf = -(10**15)
                dp = [[neg_inf] * (mh + 1) for _ in range(n_val)]
                dp[0][0] = 0
                for u, v, w in sorted((u, v, w) for u, v, w in edge_list if 0 <= u < v < n_val):
                    ew = w - (pen if v >= cn else 0)
                    for h in range(mh):
                        if dp[u][h] > neg_inf:
                            dp[v][h + 1] = max(dp[v][h + 1], dp[u][h] + ew)
                return max(0, max(max(r) for r in dp))

            def oracle(cand: str) -> bool:
                fn, status = _compile_solve_fn(cand, 2)
                if fn is None or status != "ok":
                    return False
                for n_val, elist in suites:
                    out = fn(n_val, list(elist))
                    if not isinstance(out, int) or out != ref_solve(n_val, elist):
                        return False
                return True

            def micro(cand: str) -> Tuple[bool, str]:
                fn, status = _compile_solve_fn(cand, 2)
                if fn is None:
                    return False, status
                # Smoke checks: n=0 -> 0; negative edge from 0 -> >= 0; backward edge ignored -> 0
                if fn(0, [(0, 1, 5)]) != 0:
                    return False, "invariant_violation:n_zero_must_be_zero"
                r_neg = fn(3, [(0, 1, -8), (2, 0, 99)])
                if not isinstance(r_neg, int) or r_neg != 0:
                    return False, "invariant_violation:backward_or_negative_stay"
                return True, "ok"

            return oracle, micro

        orc, mic = make_code_hard_eval(max_hops, penalty, cap_nodes)
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="code_patch",
                difficulty="hard",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    return tasks


# ---------------------------------------------------------------------------
# Domain 2 Builder: Scientific Symbolic & Dimensional Reasoning (60 tasks)
# ---------------------------------------------------------------------------


def _build_scientific_tasks() -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []

    # --- EASY (20 tasks): Reaction Kinetics Rate Constant SI Exponents [L, M, T, mol] ---
    for i in range(1, 21):
        a_ord = (i % 3) + 1
        b_ord = ((i + 1) % 2) + 1
        dim_vol = 3 if (i % 4 != 0) else 2  # 3D bulk vs 2D catalytic surface concentration!
        total_ord = a_ord + b_ord
        # [X] has units mol * L^{-dim_vol}, rate r = d[P]/dt has units mol * L^{-dim_vol} * T^{-1}
        # r = k * [A]^a * [B]^b  =>  k = r / [X]^{a+b} = (mol * L^{-d})^{1 - (a+b)} * T^{-1}
        exp_L = dim_vol * (total_ord - 1)
        exp_M = 0
        exp_T = -1
        exp_mol = 1 - total_ord
        tid = f"sci_easy_{i:02d}"
        prompt = (
            f"[Task {tid}] Determine the SI dimensional exponent vector `[L, M, T, mol]` and "
            f"overall order for rate constant `k` in `r = k [A]^{a_ord} [B]^{b_ord}` inside a "
            f"{dim_vol}D homogeneous phase (concentration `[X]` in `mol * L^-{dim_vol}`, "
            f"rate `r` in `mol * L^-{dim_vol} * s^-1`). Return JSON "
            f'`{{"order": int, "units": [L, M, T, mol]}}`.'
        )
        gt = json.dumps(
            {"order": total_ord, "units": [exp_L, exp_M, exp_T, exp_mol]}, sort_keys=True
        )
        d0 = f"order={total_ord}, units=[{exp_L}, 0, -1, {exp_mol}]"  # invalid JSON
        d1 = json.dumps(
            {"order": total_ord, "units": [exp_L, 1, +1, exp_mol]}, sort_keys=True
        )  # violates T=-1 & M=0 conservation check
        d2 = json.dumps(
            {"order": total_ord, "units": [dim_vol * total_ord, 0, -1, -total_ord]}, sort_keys=True
        )  # classic off-by-one forgetting r itself carries 1 power of concentration!

        def make_sci_easy_eval(
            ord_true: int, u_true: List[int]
        ) -> Tuple[Callable, Callable]:
            def oracle(cand: str) -> bool:
                obj = json.loads(cand)
                return (
                    isinstance(obj, dict)
                    and obj.get("order") == ord_true
                    and obj.get("units") == u_true
                )

            def micro(cand: str) -> Tuple[bool, str]:
                obj = json.loads(cand)
                if not isinstance(obj, dict) or "units" not in obj or "order" not in obj:
                    return False, "schema_missing_keys"
                u = obj["units"]
                if not (isinstance(u, list) and len(u) == 4 and all(isinstance(x, int) for x in u)):
                    return False, "invalid_units_vector"
                # Conservation microcheck: pure molar kinetics has M == 0 and T == -1
                if u[1] != 0 or u[2] != -1:
                    return False, "dimensional_conservation_violation:M_or_T"
                return True, "ok"

            return oracle, micro

        orc, mic = make_sci_easy_eval(total_ord, [exp_L, exp_M, exp_T, exp_mol])
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="scientific_symbolic",
                difficulty="easy",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    # --- MEDIUM (20 tasks): 2x2 Dynamical System ODE Jacobian Spectrum & Phase Portrait ---
    # Construct 20 distinct non-degenerate 2x2 matrices covering stable/unstable nodes, spirals, saddles
    med_matrices = [
        (-3, 1, -2, -1),   # tr=-4, det=5, disc=-4 -> stable_spiral
        (-5, 1, 2, -4),    # tr=-9, det=18, disc=9 -> stable_node
        (1, 4, 2, -1),     # tr=0, det=-9 -> saddle
        (2, -3, 3, 2),     # tr=4, det=13, disc=-36 -> unstable_spiral
        (4, 1, 2, 3),      # tr=7, det=10, disc=9 -> unstable_node
        (-2, -5, 1, -2),   # tr=-4, det=9, disc=-20 -> stable_spiral
        (-6, 2, 1, -5),    # tr=-11, det=28, disc=9 -> stable_node
        (-1, 3, 3, 1),     # tr=0, det=-10 -> saddle
        (1, -2, 4, 3),     # tr=4, det=11, disc=-28 -> unstable_spiral
        (5, 2, 1, 4),      # tr=9, det=18, disc=9 -> unstable_node
        (-4, 3, -3, -4),   # stable_spiral
        (-7, 2, 3, -6),    # stable_node
        (2, 5, 3, -2),     # saddle
        (3, -4, 2, 1),     # unstable_spiral
        (6, 1, 3, 4),      # unstable_node
        (-1, -4, 2, -3),   # stable_spiral
        (-8, 3, 2, -5),    # stable_node
        (-3, 4, 5, 2),     # saddle
        (2, -5, 2, 2),     # unstable_spiral
        (7, 2, 2, 4),      # unstable_node
    ]
    for i, (a11, a12, a21, a22) in enumerate(med_matrices, start=1):
        J = np.array([[a11, a12], [a21, a22]], dtype=np.float64)
        tr = int(a11 + a22)
        det = int(a11 * a22 - a12 * a21)
        disc = tr * tr - 4 * det
        eigs = np.linalg.eigvals(J)
        abscissa = round(float(np.max(np.real(eigs))), 4)
        if det < 0:
            cls = "saddle"
        elif disc < 0:
            cls = "stable_spiral" if tr < 0 else "unstable_spiral"
        else:
            cls = "stable_node" if tr < 0 else "unstable_node"

        tid = f"sci_med_{i:02d}"
        prompt = (
            f"[Task {tid}] Consider the linear autonomous dynamical system `dx/dt = J x` with\n"
            f"  J = [[{a11}, {a12}], [{a21}, {a22}]].\n"
            f"Compute the exact integer `trace`, integer `det`, `spectral_abscissa` "
            f"(maximum real part of eigenvalues rounded to 4 decimals), and equilibrium "
            f"`classification` in `{{stable_node, stable_spiral, saddle, unstable_node, unstable_spiral}}`.\n"
            f"Constraints: `saddle` requires `det < 0`; asymptotic stability requires "
            f"`trace < 0` and `det > 0`.\n"
            f'Return JSON `{{"classification": str, "det": int, "spectral_abscissa": float, "trace": int}}`.'
        )
        gt = json.dumps(
            {
                "classification": cls,
                "det": det,
                "spectral_abscissa": abscissa,
                "trace": tr,
            },
            sort_keys=True,
        )
        # d0: malformed JSON; d1: internal sign contradiction (claims stable_node while trace>0 or det<0);
        # d2: passes trace/det/stability sign checks, but swaps spiral <-> node (discriminant error)
        # or computes wrong eigenvalue real part on saddle!
        d0 = f"{{trace: {tr}, det: {det}, class: {cls}}}"
        contradict_cls = "stable_node" if abscissa > 0 else "saddle"
        d1 = json.dumps(
            {
                "classification": contradict_cls,
                "det": det,
                "spectral_abscissa": abscissa,
                "trace": tr,
            },
            sort_keys=True,
        )
        if cls == "stable_spiral":
            subtle_cls, subtle_abs = "stable_node", abscissa
        elif cls == "stable_node":
            subtle_cls, subtle_abs = "stable_spiral", round(tr / 2.0, 4)
        elif cls == "unstable_spiral":
            subtle_cls, subtle_abs = "unstable_node", abscissa
        elif cls == "unstable_node":
            subtle_cls, subtle_abs = "unstable_spiral", round(tr / 2.0, 4)
        else:  # saddle
            subtle_cls, subtle_abs = "saddle", round(abscissa + 0.75, 4)
        d2 = json.dumps(
            {
                "classification": subtle_cls,
                "det": det,
                "spectral_abscissa": subtle_abs,
                "trace": tr,
            },
            sort_keys=True,
        )

        def make_sci_med_eval(
            tr_t: int, det_t: int, abs_t: float, cls_t: str
        ) -> Tuple[Callable, Callable]:
            valid_labels = {
                "stable_node",
                "stable_spiral",
                "saddle",
                "unstable_node",
                "unstable_spiral",
            }

            def oracle(cand: str) -> bool:
                obj = json.loads(cand)
                return (
                    isinstance(obj, dict)
                    and obj.get("trace") == tr_t
                    and obj.get("det") == det_t
                    and obj.get("classification") == cls_t
                    and abs(float(obj.get("spectral_abscissa", 1e9)) - abs_t) <= 1e-3
                )

            def micro(cand: str) -> Tuple[bool, str]:
                obj = json.loads(cand)
                if not isinstance(obj, dict):
                    return False, "not_dict"
                c_lbl = obj.get("classification")
                d_val = obj.get("det")
                sa_val = obj.get("spectral_abscissa")
                if c_lbl not in valid_labels or not isinstance(d_val, int) or not isinstance(
                    sa_val, (int, float)
                ):
                    return False, "schema_invalid"
                # Public algebraic consistency check between classification, det sign, and abscissa sign
                if c_lbl == "saddle" and not (d_val < 0 and sa_val > 0):
                    return False, "consistency_violation:saddle_requires_neg_det_pos_abscissa"
                if c_lbl.startswith("stable") and not (d_val > 0 and sa_val < 0):
                    return False, "consistency_violation:stable_requires_pos_det_neg_abscissa"
                if c_lbl.startswith("unstable") and not (d_val > 0 and sa_val > 0):
                    return False, "consistency_violation:unstable_requires_pos_det_pos_abscissa"
                return True, "ok"

            return oracle, micro

        orc, mic = make_sci_med_eval(tr, det, abscissa, cls)
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="scientific_symbolic",
                difficulty="medium",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    # --- HARD (20 tasks): Schnakenberg Reaction-Diffusion Linearization & Turing Dispersion ---
    # Reaction kinetics: du/dt = a - u + u^2 v + d_u lap(u), dv/dt = b - u^2 v + d_v lap(v)
    # Homogeneous positive equilibrium: u* = a + b, v* = b / (a + b)^2
    # At wavenumber k, dispersion matrix M(k) = J(u*, v*) - k^2 diag(d_u, d_v)
    # where J_11 = 2 u* v* - 1 = (2b)/(a+b) - 1 = (b - a)/(a + b), J_12 = (u*)^2 = (a+b)^2,
    #       J_21 = -2 u* v* = -2b/(a+b), J_22 = -(u*)^2 = -(a+b)^2.
    for i in range(1, 21):
        a_param = 0.2 + 0.05 * (i % 5)
        b_param = 0.9 + 0.12 * (i % 6)
        du = 1.0
        dv = 8.0 + 2.5 * (i % 7)
        k_wave = 0.6 + 0.15 * (i % 5)

        u_star = a_param + b_param
        j11 = (b_param - a_param) / u_star
        j12 = u_star * u_star
        j21 = -2.0 * b_param / u_star
        j22 = -(u_star * u_star)

        k2 = k_wave * k_wave
        M_k = np.array([[j11 - du * k2, j12], [j21, j22 - dv * k2]], dtype=np.float64)
        eigs_k = np.linalg.eigvals(M_k)
        disp_abscissa = round(float(np.max(np.real(eigs_k))), 4)
        frob_sq = round(float(np.sum(M_k * M_k)), 4)
        regime = "turing_unstable" if disp_abscissa > 0.0 else "dispersion_stable"

        tid = f"sci_hard_{i:02d}"
        prompt = (
            f"[Task {tid}] Analyze the coupled reaction-diffusion PDE system (Schnakenberg kinetics):\n"
            f"  du/dt = a - u + u^2 v + d_u * nabla^2 u\n"
            f"  dv/dt = b - u^2 v + d_v * nabla^2 v\n"
            f"with parameters `a = {a_param:.2f}`, `b = {b_param:.2f}`, `d_u = {du:.1f}`, "
            f"`d_v = {dv:.1f}` perturbed at Fourier wavenumber `k = {k_wave:.2f}`.\n"
            f"Step 1 (Equilibrium Invariant): Compute unique spatially uniform equilibrium `(u^*, v^*)`.\n"
            f"Step 2 (Jacobian & Diffusion Operator): Form `M(k) = J(u^*, v^*) - k^2 diag(d_u, d_v)`.\n"
            f"Step 3 (Spectral Diagnostics): Compute `u_star` (4 dec), `frob_norm_sq = ||M(k)||_F^2` (4 dec), "
            f"`dispersion_abscissa = max Re eig(M(k))` (4 dec), and `regime` in "
            f"`{{turing_unstable, dispersion_stable}}` (strictly unstable iff `dispersion_abscissa > 0`).\n"
            f'Return JSON `{{"dispersion_abscissa": float, "frob_norm_sq": float, "regime": str, "u_star": float}}`.'
        )
        gt = json.dumps(
            {
                "dispersion_abscissa": disp_abscissa,
                "frob_norm_sq": frob_sq,
                "regime": regime,
                "u_star": round(u_star, 4),
            },
            sort_keys=True,
        )
        # d0: syntax error; d1: negative frob_norm_sq or sign mismatch between abscissa and regime;
        # d2: uses k instead of k^2 in M(k) = J - k diag(d_u, d_v), keeping regime internally consistent
        # with its miscomputed abscissa so microcheck passes while oracle catches the k vs k^2 error!
        M_wrong = np.array(
            [[j11 - du * k_wave, j12], [j21, j22 - dv * k_wave]], dtype=np.float64
        )
        wrong_abs = round(float(np.max(np.real(np.linalg.eigvals(M_wrong)))), 4)
        wrong_frob = round(float(np.sum(M_wrong * M_wrong)), 4)
        wrong_reg = "turing_unstable" if wrong_abs > 0.0 else "dispersion_stable"

        d0 = "not_valid_json_output"
        d1 = json.dumps(
            {
                "dispersion_abscissa": disp_abscissa,
                "frob_norm_sq": -abs(frob_sq),
                "regime": "turing_unstable" if disp_abscissa <= 0 else "dispersion_stable",
                "u_star": round(u_star, 4),
            },
            sort_keys=True,
        )
        d2 = json.dumps(
            {
                "dispersion_abscissa": wrong_abs,
                "frob_norm_sq": wrong_frob,
                "regime": wrong_reg,
                "u_star": round(u_star, 4),
            },
            sort_keys=True,
        )

        def make_sci_hard_eval(
            u_t: float, f_t: float, a_t: float, r_t: str
        ) -> Tuple[Callable, Callable]:
            def oracle(cand: str) -> bool:
                obj = json.loads(cand)
                return (
                    isinstance(obj, dict)
                    and obj.get("regime") == r_t
                    and abs(float(obj.get("u_star", -999.0)) - u_t) <= 1e-3
                    and abs(float(obj.get("frob_norm_sq", -999.0)) - f_t) <= 1e-2
                    and abs(float(obj.get("dispersion_abscissa", -999.0)) - a_t) <= 1e-3
                )

            def micro(cand: str) -> Tuple[bool, str]:
                obj = json.loads(cand)
                if not isinstance(obj, dict):
                    return False, "not_dict"
                reg = obj.get("regime")
                fn_sq = obj.get("frob_norm_sq")
                da = obj.get("dispersion_abscissa")
                us = obj.get("u_star")
                if (
                    reg not in ("turing_unstable", "dispersion_stable")
                    or not isinstance(fn_sq, (int, float))
                    or not isinstance(da, (int, float))
                    or not isinstance(us, (int, float))
                ):
                    return False, "schema_invalid"
                if fn_sq <= 0.0 or us <= 0.0:
                    return False, "positivity_invariant_violation"
                if (da > 0.0 and reg != "turing_unstable") or (
                    da <= 0.0 and reg != "dispersion_stable"
                ):
                    return False, "regime_sign_contradiction"
                return True, "ok"

            return oracle, micro

        orc, mic = make_sci_hard_eval(round(u_star, 4), frob_sq, disp_abscissa, regime)
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="scientific_symbolic",
                difficulty="hard",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    return tasks


# ---------------------------------------------------------------------------
# Domain 3 Builder: Multi-Constraint Evidence Cross-Checking (60 tasks)
# ---------------------------------------------------------------------------


def _build_evidence_tasks() -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []

    # --- EASY (20 tasks): Two-Table Timestamp Reconciliation + 3-Job Slot Assignment ---
    for i in range(1, 21):
        val_primary = 120 + 5 * i
        val_stale = val_primary + 15
        blocked_slot = i % 3
        # Assign jobs J0, J1, J2 to slots {0, 1, 2}: J0 cannot use blocked_slot, J1 < J2
        rem = [s for s in (0, 1, 2) if s != blocked_slot]
        # Lexicographically smallest valid slot assignment [s_J0, s_J1, s_J2]
        valid_assigns = [
            [s0, s1, s2]
            for s0 in range(3)
            for s1 in range(3)
            for s2 in range(3)
            if len({s0, s1, s2}) == 3 and s0 != blocked_slot and s1 < s2
        ]
        best_sched = sorted(valid_assigns)[0]
        tid = f"evid_easy_{i:02d}"
        prompt = (
            f"[Task {tid}] Reconcile `metric_mw` from two telemetry tables (latest timestamp wins) "
            f"and schedule `[J0, J1, J2]` onto distinct slots `0, 1, 2`:\n"
            f"  | Source_Replica | ts=1040 | metric_mw={val_stale} |\n"
            f"  | Source_Audit   | ts=1095 | metric_mw={val_primary} |\n"
            f"Constraints: `J0 != {blocked_slot}` and `slot(J1) < slot(J2)` (pick lexicographically "
            f"smallest valid `[s0, s1, s2]`). Return JSON "
            f'`{{"metric_mw": int, "schedule": [s0, s1, s2]}}`.'
        )
        gt = json.dumps({"metric_mw": val_primary, "schedule": best_sched}, sort_keys=True)
        d0 = f"metric={val_primary}, schedule={best_sched}"
        d1 = json.dumps(
            {"metric_mw": val_primary, "schedule": [blocked_slot, rem[0], rem[0]]}, sort_keys=True
        )  # duplicate slot & blocked violation
        d2 = json.dumps(
            {"metric_mw": val_stale, "schedule": best_sched}, sort_keys=True
        )  # valid schedule (passes microcheck!), stale table value!

        def make_evid_easy_eval(
            v_true: int, sched_true: List[int], blk: int
        ) -> Tuple[Callable, Callable]:
            def oracle(cand: str) -> bool:
                obj = json.loads(cand)
                return (
                    isinstance(obj, dict)
                    and obj.get("metric_mw") == v_true
                    and obj.get("schedule") == sched_true
                )

            def micro(cand: str) -> Tuple[bool, str]:
                obj = json.loads(cand)
                if not isinstance(obj, dict) or "schedule" not in obj or "metric_mw" not in obj:
                    return False, "schema_invalid"
                sc = obj["schedule"]
                if not (isinstance(sc, list) and sorted(sc) == [0, 1, 2]):
                    return False, "schedule_not_permutation"
                if sc[0] == blk or not (sc[1] < sc[2]):
                    return False, "public_schedule_constraint_violated"
                return True, "ok"

            return oracle, micro

        orc, mic = make_evid_easy_eval(val_primary, best_sched, blocked_slot)
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="evidence_crosscheck",
                difficulty="easy",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    # --- MEDIUM (20 tasks): Four-Source Authority/Checksum Ledger + 4-Task Precedence DAG ---
    for i in range(1, 21):
        base = 250 + 10 * i
        # Authority: Tier-1 Signed_Audit (if checksum=="VALID") > Tier-2 Sensor_Bus > Tier-3 Cache
        # Here Source_A is Tier-1 but checksum=="CORRUPT"! So Tier-2 latest valid Source_C wins!
        val_corrupt_t1 = base + 90
        val_old_t2 = base - 20
        val_latest_t2 = base + 5
        val_cache = base + 45
        forbid_t0 = i % 4
        # 4 tasks T0..T3 onto slots 0..3: precedence T0 < T2, T1 < T3, mutex T0 != forbid_t0
        valid_4 = [
            [s0, s1, s2, s3]
            for s0 in range(4)
            for s1 in range(4)
            for s2 in range(4)
            for s3 in range(4)
            if len({s0, s1, s2, s3}) == 4 and s0 < s2 and s1 < s3 and s0 != forbid_t0
        ]
        best_4 = sorted(valid_4)[0]
        tid = f"evid_med_{i:02d}"
        prompt = (
            f"[Task {tid}] Cross-check four conflicting ledger records under strict provenance rules:\n"
            f"  | Source_A | tier=1 | ts=1850 | checksum=CORRUPT | net_kWh={val_corrupt_t1} |\n"
            f"  | Source_B | tier=2 | ts=1710 | checksum=VALID   | net_kWh={val_old_t2} |\n"
            f"  | Source_C | tier=2 | ts=1820 | checksum=VALID   | net_kWh={val_latest_t2} |\n"
            f"  | Source_D | tier=3 | ts=1890 | checksum=VALID   | net_kWh={val_cache} |\n"
            f"Provenance Rules:\n"
            f"  - Discard any record with `checksum=CORRUPT` regardless of authority tier or timestamp.\n"
            f"  - Among valid records, select lowest authority tier number (`tier=1` highest priority), "
            f"breaking ties by maximum timestamp `ts`.\n"
            f"Scheduling Constraints for `[T0, T1, T2, T3]` onto distinct slots `0..3`:\n"
            f"  - Precedence: `slot(T0) < slot(T2)` and `slot(T1) < slot(T3)`.\n"
            f"  - Mutex: `slot(T0) != {forbid_t0}`. Pick lexicographically smallest valid schedule.\n"
            f'Return JSON `{{"invalid_sources": ["Source_A"], "net_kWh": int, "schedule": [s0,s1,s2,s3]}}`.'
        )
        gt = json.dumps(
            {
                "invalid_sources": ["Source_A"],
                "net_kWh": val_latest_t2,
                "schedule": best_4,
            },
            sort_keys=True,
        )
        d0 = "invalid_json"
        d1 = json.dumps(
            {
                "invalid_sources": ["Source_A"],
                "net_kWh": val_latest_t2,
                "schedule": [3, 2, 1, 0],  # violates precedence T0 < T2!
            },
            sort_keys=True,
        )
        d2 = json.dumps(
            {
                "invalid_sources": [],
                "net_kWh": val_corrupt_t1,  # fooled by tier=1 without filtering CORRUPT checksum, or picked Source_D!
                "schedule": best_4,
            },
            sort_keys=True,
        )

        def make_evid_med_eval(
            v_true: int, sc_true: List[int], fb: int
        ) -> Tuple[Callable, Callable]:
            def oracle(cand: str) -> bool:
                obj = json.loads(cand)
                return (
                    isinstance(obj, dict)
                    and obj.get("net_kWh") == v_true
                    and obj.get("schedule") == sc_true
                    and obj.get("invalid_sources") == ["Source_A"]
                )

            def micro(cand: str) -> Tuple[bool, str]:
                obj = json.loads(cand)
                if not isinstance(obj, dict) or "schedule" not in obj or "net_kWh" not in obj:
                    return False, "schema_invalid"
                sc = obj["schedule"]
                if not (isinstance(sc, list) and sorted(sc) == [0, 1, 2, 3]):
                    return False, "not_permutation_4"
                if not (sc[0] < sc[2] and sc[1] < sc[3] and sc[0] != fb):
                    return False, "precedence_or_mutex_violation"
                return True, "ok"

            return oracle, micro

        orc, mic = make_evid_med_eval(val_latest_t2, best_4, forbid_t0)
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="evidence_crosscheck",
                difficulty="medium",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    # --- HARD (20 tasks): Six-Source Unit-Normalized Cross-Check + 5-Worker Coupled Quota/DAG ---
    for i in range(1, 21):
        mw_val = 4 + (i % 5)  # reported in MW in Source_S4 -> must normalize to kW (* 1000)!
        kw_offset = 150 + 10 * i
        true_kw = mw_val * 1000 + kw_offset
        stale_kw = true_kw - 600
        corrupt_kw = true_kw + 1200
        unverified_kw = mw_val + kw_offset  # classic unit normalization error (forgot MW -> kW!)

        # 5 jobs W0..W4 assigned to distinct time slots 0..4:
        # Constraints:
        #   1) Precedence chain: W0 < W2 < W4
        #   2) Precedence pair: W1 < W3
        #   3) Separation mutex: abs(slot(W0) - slot(W1)) >= 2
        #   4) Window cap: slot(W3) <= 4 and slot(W0) != (i % 2)
        ban_w0 = i % 2
        valid_5 = [
            [s0, s1, s2, s3, s4]
            for s0 in range(5)
            for s1 in range(5)
            for s2 in range(5)
            for s3 in range(5)
            for s4 in range(5)
            if (
                len({s0, s1, s2, s3, s4}) == 5
                and s0 < s2 < s4
                and s1 < s3
                and abs(s0 - s1) >= 2
                and s0 != ban_w0
            )
        ]
        # Wait: let's guarantee valid_5 is non-empty! Check:
        # If ban_w0 == 1: s0=0 -> s2 in {1,2,3}, s4 in {2,3,4}; abs(0 - s1) >= 2 -> s1 in {2,3}; s1 < s3!
        # e.g. s0=0, s1=2, s2=1, s3=3, s4=4 -> 0 < 1 < 4 (True!), 2 < 3 (True!), |0-2|=2 >= 2 (True!).
        # If ban_w0 == 0: s0=1 -> s2 in {2,3}, s4=4; abs(1 - s1) >= 2 -> s1=3? Then s3=4 collides with s4=4!
        # Wait: if s0=1 and abs(1 - s1) >= 2, could s1=0? Wait: |1 - 0| = 1 < 2! If s1=3, s3=4, then s4 must be > s2 > s0=1, so s4>=3, collision with {3,4}!
        # Aha! Catching that mathematically BEFORE running is why careful verification rocks!
        # If ban_w0 == 0, let's check s0=2 -> s2=3, s4=4, s1=0, s3=1!
        # Check [2, 0, 3, 1, 4]:
        #   unique: {0,1,2,3,4} (True!)
        #   s0 < s2 < s4: 2 < 3 < 4 (True!)
        #   s1 < s3: 0 < 1 (True!)
        #   |s0 - s1| = |2 - 0| = 2 >= 2 (True!)
        #   s0 != 0: 2 != 0 (True!)
        # BOTH ban_w0=0 and ban_w0=1 have valid solutions!!
        best_5 = sorted(valid_5)[0]

        tid = f"evid_hard_{i:02d}"
        prompt = (
            f"[Task {tid}] Synthesize reconciled facility load (`total_kW` in kW) across six "
            f"conflicting multi-department tables and solve the 5-stage maintenance DAG:\n"
            f"  | Source_S1 | dept=Ops   | tier=1 | ts=2100 | crc=FAIL | load={corrupt_kw} kW |\n"
            f"  | Source_S2 | dept=Grid  | tier=2 | ts=1980 | crc=PASS | load={stale_kw} kW |\n"
            f"  | Source_S3 | dept=Meter | tier=1 | ts=2050 | crc=FAIL | base={mw_val + 2} MW |\n"
            f"  | Source_S4 | dept=Grid  | tier=2 | ts=2140 | crc=PASS | base={mw_val} MW |\n"
            f"  | Source_S5 | dept=Aux   | tier=2 | ts=2140 | crc=PASS | aux_delta={kw_offset} kW |\n"
            f"  | Source_S6 | dept=Cache | tier=3 | ts=2190 | crc=PASS | load={stale_kw + 95} kW |\n"
            f"Cross-Verification & Unit Normalization Rules:\n"
            f"  - Rule 1 (CRC Integrity): Exclude all rows with `crc=FAIL` (`Source_S1`, `Source_S3`).\n"
            f"  - Rule 2 (Authority & Co-Timestamp Synthesis): Highest valid authority is `tier=2` "
            f"at maximum timestamp `ts=2140`, which splits across primary bus `Source_S4` (in **MW**) "
            f"and auxiliary bus `Source_S5` (in **kW**).\n"
            f"  - Rule 3 (Dimensional Conversion): Convert `base` from MW to kW (`1 MW = 1000 kW`) "
            f"and add `aux_delta` to obtain integer `total_kW`.\n"
            f"Coupled 5-Worker Scheduling Constraints (`[W0, W1, W2, W3, W4]` -> permutation of `0..4`):\n"
            f"  - Rule 4 (Chain Precedence): `slot(W0) < slot(W2) < slot(W4)` and `slot(W1) < slot(W3)`.\n"
            f"  - Rule 5 (Mutex Separation): `abs(slot(W0) - slot(W1)) >= 2` and `slot(W0) != {ban_w0}`.\n"
            f"  - Rule 6 (Canonical Tie-Break): Select lexicographically smallest valid slot vector.\n"
            f'Return JSON `{{"excluded": ["Source_S1", "Source_S3"], "schedule": [s0,s1,s2,s3,s4], "total_kW": int}}`.'
        )
        gt = json.dumps(
            {
                "excluded": ["Source_S1", "Source_S3"],
                "schedule": best_5,
                "total_kW": true_kw,
            },
            sort_keys=True,
        )
        d0 = "{broken_json"
        d1 = json.dumps(
            {
                "excluded": ["Source_S1", "Source_S3"],
                "schedule": [0, 1, 2, 3, 4],  # |s0 - s1| = 1 < 2 -> fails mutex separation microcheck!
                "total_kW": true_kw,
            },
            sort_keys=True,
        )
        d2 = json.dumps(
            {
                "excluded": ["Source_S1", "Source_S3"],
                "schedule": best_5,  # passes all public schedule & schema microchecks!
                "total_kW": unverified_kw,  # unit conversion error (forgot *1000 on MW)!
            },
            sort_keys=True,
        )

        def make_evid_hard_eval(
            kw_true: int, sc5_true: List[int], bw0: int
        ) -> Tuple[Callable, Callable]:
            def oracle(cand: str) -> bool:
                obj = json.loads(cand)
                return (
                    isinstance(obj, dict)
                    and obj.get("total_kW") == kw_true
                    and obj.get("schedule") == sc5_true
                    and obj.get("excluded") == ["Source_S1", "Source_S3"]
                )

            def micro(cand: str) -> Tuple[bool, str]:
                obj = json.loads(cand)
                if not isinstance(obj, dict) or "schedule" not in obj or "total_kW" not in obj:
                    return False, "schema_invalid"
                sc = obj["schedule"]
                if not (isinstance(sc, list) and sorted(sc) == [0, 1, 2, 3, 4]):
                    return False, "not_permutation_5"
                s0, s1, s2, s3, s4 = sc
                if not (s0 < s2 < s4 and s1 < s3 and abs(s0 - s1) >= 2 and s0 != bw0):
                    return False, "coupled_dag_or_mutex_violation"
                return True, "ok"

            return oracle, micro

        orc, mic = make_evid_hard_eval(true_kw, best_5, ban_w0)
        tasks.append(
            BenchmarkTask(
                task_id=tid,
                domain="evidence_crosscheck",
                difficulty="hard",
                prompt=prompt,
                ground_truth=gt,
                distractors=(d0, d1, d2),
                _oracle_fn=orc,
                _microcheck_fn=mic,
            )
        )

    return tasks


def build_benchmark_suite() -> List[BenchmarkTask]:
    """Construct the complete 180-task verifiable CrossVerify-Bench suite."""
    suite: List[BenchmarkTask] = []
    suite.extend(_build_code_patch_tasks())
    suite.extend(_build_scientific_tasks())
    suite.extend(_build_evidence_tasks())
    return suite


def summarize_benchmark_suite(tasks: Sequence[BenchmarkTask]) -> Dict[str, Any]:
    """Return structural statistics and domain/difficulty counts of the benchmark suite."""
    by_domain: Dict[str, int] = {}
    by_diff: Dict[str, int] = {}
    scores_by_diff: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}

    for t in tasks:
        by_domain[t.domain] = by_domain.get(t.domain, 0) + 1
        by_diff[t.difficulty] = by_diff.get(t.difficulty, 0) + 1
        feats = extract_prompt_features(t.prompt)
        scores_by_diff[t.difficulty].append(feats.complexity_score)

    return {
        "total_tasks": len(tasks),
        "by_domain": by_domain,
        "by_difficulty": by_diff,
        "mean_prompt_complexity_by_difficulty": {
            k: float(np.mean(v)) if v else 0.0 for k, v in scores_by_diff.items()
        },
    }
