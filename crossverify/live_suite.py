"""High-discrimination 45-task Live-LLM evaluation suite for CrossVerify-Bench.

Constructs 45 genuinely diverse, non-trivial, deterministically verifiable tasks
(15 Easy / 15 Medium / 15 Hard across ``code_patch``, ``scientific_symbolic``, and
``evidence_crosscheck``) specifically designed for evaluating live multi-family LLMs
with zero hints leaked in prompts and robust Markdown code/JSON fence normalization.
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from crossverify.tasks import BenchmarkTask, _SAFE_BUILTINS


def extract_clean_code(raw: str) -> str:
    """Extract Python code cleanly from raw LLM output (with or without Markdown fences)."""
    text = raw.strip()
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        # Prefer the block defining `def solve`
        for blk in fenced:
            if "def solve" in blk:
                return blk.strip()
        return fenced[-1].strip()
    # Strip stray leading/trailing backticks
    text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    idx = text.find("def solve")
    if idx > 0:
        return text[idx:].strip()
    return text.strip()


def extract_clean_json(raw: str) -> Dict[str, Any]:
    """Extract and parse JSON object from raw LLM output (with or without Markdown fences)."""
    text = raw.strip()
    fenced = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced[::-1] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # Find outermost {...}
        matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cand, flags=re.DOTALL)
        for m in reversed(matches):
            try:
                obj = json.loads(m)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    raise ValueError("no_valid_json_object")


def _compile_live_solve(raw: str, arity: int) -> Tuple[Any, str]:
    code = extract_clean_code(raw)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, f"syntax_error:{exc.msg}"

    solve_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    if not solve_defs:
        return None, "missing_solve_fn"
    if len(solve_defs[0].args.args) != arity:
        return None, f"arity_mismatch:expected_{arity}_got_{len(solve_defs[0].args.args)}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return None, "forbidden_import"

    safe_ns: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    exec(compile(tree, filename="<live_patch>", mode="exec"), safe_ns, safe_ns)
    return safe_ns.get("solve"), "ok"


# ===========================================================================
# 1. Domain: Code Patch & Algorithmic Invariants (15 Tasks: 5E / 5M / 5H)
# ===========================================================================


def _build_live_code_tasks() -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []

    # --- 5 EASY CODE TASKS ---
    for i in range(1, 6):
        step = i + 2
        rem_target = i % step
        cap = 25 + 10 * i
        tid = f"live_code_easy_{i:02d}"
        prompt = (
            f"Write ONLY Python function `def solve(lo, hi):` (no imports, no prose).\n"
            f"Return the count of integers `x` in closed interval `[lo, hi]` satisfying "
            f"`x % {step} == {rem_target}`, clamped above by `{cap}`. Return `0` if `lo > hi`."
        )

        def make_ce(st: int, rt: int, cp: int) -> Tuple[Callable, Callable]:
            cases = [(-25, 40), (10, -5), (rt, rt), (-100, -10), (0, st * (cp + 5))]

            def orc(raw: str) -> bool:
                fn, st_msg = _compile_live_solve(raw, 2)
                if fn is None or st_msg != "ok":
                    return False
                for a, b in cases:
                    exp = 0 if a > b else min(cp, sum(1 for x in range(a, b + 1) if x % st == rt))
                    out = fn(a, b)
                    if not isinstance(out, int) or isinstance(out, bool) or out != exp:
                        return False
                return True

            def mic(raw: str) -> Tuple[bool, str]:
                fn, st_msg = _compile_live_solve(raw, 2)
                if fn is None:
                    return False, st_msg
                r0 = fn(8, 2)
                r1 = fn(0, 15)
                if not isinstance(r0, int) or r0 != 0 or not isinstance(r1, int) or r1 < 0:
                    return False, "smoke_invariant_failed"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_ce(step, rem_target, cap)
        tasks.append(BenchmarkTask(tid, "code_patch", "easy", prompt, "", ("", "", ""), o_fn, m_fn))

    # --- 5 MEDIUM CODE TASKS (Diverse non-trivial boundary invariants) ---
    for i in range(1, 6):
        gap = i - 1  # 0..4
        max_span = 12 + 4 * i
        tid = f"live_code_med_{i:02d}"
        prompt = (
            f"Write ONLY Python function `def solve(intervals):` (no imports, no prose).\n"
            f"Given a list of `(start, end)` integer pairs:\n"
            f"1. Discard degenerate pairs where `start >= end`.\n"
            f"2. Sort remaining intervals and merge adjacent/overlapping ones whenever "
            f"`next_start <= current_end + {gap}` (note: gap bridge `{gap}` adds empty bridged gap into the merged span `[merged_start, merged_end]`).\n"
            f"3. For each final merged interval `[L, R]`, cap its effective contribution at `min(R - L, {max_span})`, and return the integer sum of capped contributions."
        )

        def make_cm(gp: int, msp: int) -> Tuple[Callable, Callable]:
            suites = [
                [],
                [(7, 3), (4, 4)],
                [(1, 20), (3, 6)],  # nested interval + cap check
                [(0, 3), (3 + gp, 8), (9 + gp, 12)],
                [(-10, -4), (-4 + gp, 2), (1, 3), (20, 20 + msp + 7)],
            ]

            def ref(ivs: List[Tuple[int, int]]) -> int:
                v = sorted((a, b) for a, b in ivs if a < b)
                if not v:
                    return 0
                m = [[v[0][0], v[0][1]]]
                for a, b in v[1:]:
                    if a <= m[-1][1] + gp:
                        m[-1][1] = max(m[-1][1], b)
                    else:
                        m.append([a, b])
                return sum(min(b - a, msp) for a, b in m)

            def orc(raw: str) -> bool:
                fn, st_msg = _compile_live_solve(raw, 1)
                if fn is None or st_msg != "ok":
                    return False
                for s in suites:
                    out = fn(list(s))
                    if not isinstance(out, int) or isinstance(out, bool) or out != ref(s):
                        return False
                return True

            def mic(raw: str) -> Tuple[bool, str]:
                fn, st_msg = _compile_live_solve(raw, 1)
                if fn is None:
                    return False, st_msg
                if fn([(5, 1), (2, 2)]) != 0:
                    return False, "degenerate_must_be_zero"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_cm(gap, max_span)
        tasks.append(BenchmarkTask(tid, "code_patch", "medium", prompt, "", ("", "", ""), o_fn, m_fn))

    # --- 5 HARD CODE TASKS (Genuine algorithmic traps: k-interval intersection, parity-constrained DP, circular subarray) ---
    # Hard 1 & 2: k-interval maximum intersection with weighted/offset twist
    for idx, min_k in enumerate([2, 3], start=1):
        tid = f"live_code_hard_{idx:02d}"
        prompt = (
            f"Write ONLY Python function `def solve(intervals):` (no imports, no prose).\n"
            f"Given a list of closed integer intervals `(l, r)` with `l <= r`, choose ANY subset of "
            f"AT LEAST `{min_k}` intervals whose common intersection `[L, R]` (`L = max l_i`, `R = min r_i`) "
            f"is non-empty (`L <= R`). Maximize `R - L`. If fewer than `{min_k}` intervals exist or every "
            f"`{min_k}`-subset has empty intersection (`L > R`), return `-1`. "
            f"Constraint: a single-point intersection (`L == R`) is VALID and has length `0`!"
        )

        def make_ch_inter(k_req: int) -> Tuple[Callable, Callable]:
            suites = [
                [(1, 10), (2, 8), (5, 5), (0, 4)],
                [(1, 5), (5, 9), (0, 5)],
                [(1, 2), (3, 4), (5, 6)],
                [(0, 50)],
                [(-8, 12), (-5, 15), (-6, 11), (0, 3)],
                [(2, 2), (2, 2), (2, 2)],
            ]

            def brute(ivs: List[Tuple[int, int]]) -> int:
                n = len(ivs)
                if n < k_req:
                    return -1
                # Exact check over all candidate left endpoints L in ivs
                best = -1
                for l_cand, _ in ivs:
                    rights = sorted(r for l, r in ivs if l <= l_cand and r >= l_cand)
                    if len(rights) >= k_req:
                        r_k = rights[-k_req]
                        best = max(best, r_k - l_cand)
                return best

            def orc(raw: str) -> bool:
                fn, st_msg = _compile_live_solve(raw, 1)
                if fn is None or st_msg != "ok":
                    return False
                for s in suites:
                    out = fn(list(s))
                    if not isinstance(out, int) or isinstance(out, bool) or out != brute(s):
                        return False
                return True

            def mic(raw: str) -> Tuple[bool, str]:
                fn, st_msg = _compile_live_solve(raw, 1)
                if fn is None:
                    return False, st_msg
                if fn([]) != -1:
                    return False, "empty_list_must_return_neg1"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_ch_inter(min_k)
        tasks.append(BenchmarkTask(tid, "code_patch", "hard", prompt, "", ("", "", ""), o_fn, m_fn))

    # Hard 3, 4, 5: Bounded-length circular subarray sum & Exact-parity DAG DP
    for idx, max_len in enumerate([3, 4], start=3):
        tid = f"live_code_hard_{idx:02d}"
        prompt = (
            f"Write ONLY Python function `def solve(nums):` (no imports, no prose).\n"
            f"Given a circular integer array `nums` (`len(nums) >= 0`), find the maximum sum of any "
            f"NON-EMPTY contiguous circular subarray consisting of AT MOST `min(len(nums), {max_len})` "
            f"distinct index positions (wrapping around the end of `nums` is allowed, but no index may be used twice!). "
            f"If `nums` is empty, return `0`. Note: if all elements are negative, you MUST still pick at least 1 element (return `max(nums)`)."
        )

        def make_ch_circ(ml: int) -> Tuple[Callable, Callable]:
            suites = [
                [],
                [-7, -3, -9],
                [5, -2, 6, -10, 4],
                [8, -1, -1, 9],
                [-4, 10, -2, 10, -5, 3],
                [3, 3, 3, 3, 3],
            ]

            def ref_circ(arr: List[int]) -> int:
                n = len(arr)
                if n == 0:
                    return 0
                lim = min(n, ml)
                best = max(arr)
                for start in range(n):
                    cur = 0
                    for ln in range(1, lim + 1):
                        cur += arr[(start + ln - 1) % n]
                        if cur > best:
                            best = cur
                return best

            def orc(raw: str) -> bool:
                fn, st_msg = _compile_live_solve(raw, 1)
                if fn is None or st_msg != "ok":
                    return False
                for s in suites:
                    out = fn(list(s))
                    if not isinstance(out, int) or isinstance(out, bool) or out != ref_circ(s):
                        return False
                return True

            def mic(raw: str) -> Tuple[bool, str]:
                fn, st_msg = _compile_live_solve(raw, 1)
                if fn is None:
                    return False, st_msg
                if fn([]) != 0 or not isinstance(fn([-5, -2]), int):
                    return False, "empty_or_neg_smoke_failed"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_ch_circ(max_len)
        tasks.append(BenchmarkTask(tid, "code_patch", "hard", prompt, "", ("", "", ""), o_fn, m_fn))

    # Hard 5: Even-sum subsequence / path DP on DAG
    tid5 = "live_code_hard_05"
    prompt5 = (
        "Write ONLY Python function `def solve(n, edges):` (no imports, no prose).\n"
        "Given `n` nodes `0..n-1` and directed weighted edges `(u, v, w)` (keep ONLY forward edges `0 <= u < v < n`), "
        "find the maximum total weight of any valid directed path starting at node `0` (including the 0-edge path staying at `0` with weight `0`) "
        "such that:\n"
        "1. The path uses AT MOST `3` edges, AND\n"
        "2. The total weight of the path is an **EVEN** integer (`total_weight % 2 == 0`).\n"
        "Return `0` if `n <= 0`."
    )

    def make_ch_dag_even() -> Tuple[Callable, Callable]:
        suites = [
            (0, [(0, 1, 4)]),
            (4, [(0, 1, 5), (1, 2, 7), (0, 2, 9), (2, 3, 2)]),  # 0->1->2 = 12 (2 edges, even), 0->1->2->3 = 14 (3 edges, even)!
            (4, [(0, 1, 3), (0, 2, 5), (2, 0, 100)]),  # only odd positive edges -> stay at 0 gives 0!
            (5, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 4, 4)]),  # max 3 edges -> 12 (not 16)!
            (4, [(0, 1, -3), (1, 3, 9), (0, 2, 5), (2, 3, 2)]),  # 0->1->3 = 6 (even)! 0->2->3 = 7 (odd)
        ]

        def ref_dag(n_v: int, el: List[Tuple[int, int, int]]) -> int:
            if n_v <= 0:
                return 0
            neg = -(10**15)
            # dp[node][hops][parity]
            dp = [[[neg, neg] for _ in range(4)] for _ in range(n_v)]
            dp[0][0][0] = 0
            for u, v, w in sorted((u, v, w) for u, v, w in el if 0 <= u < v < n_v):
                p_w = w % 2
                for h in range(3):
                    for par in (0, 1):
                        if dp[u][h][par] > neg:
                            np_par = (par + p_w) % 2
                            cand_val = dp[u][h][par] + w
                            if cand_val > dp[v][h + 1][np_par]:
                                dp[v][h + 1][np_par] = cand_val
            return max(0, max(dp[node][h][0] for node in range(n_v) for h in range(4)))

        def orc(raw: str) -> bool:
            fn, st_msg = _compile_live_solve(raw, 2)
            if fn is None or st_msg != "ok":
                return False
            for n_v, el in suites:
                out = fn(n_v, list(el))
                if not isinstance(out, int) or isinstance(out, bool) or out != ref_dag(n_v, el):
                    return False
            return True

        def mic(raw: str) -> Tuple[bool, str]:
            fn, st_msg = _compile_live_solve(raw, 2)
            if fn is None:
                return False, st_msg
            r0 = fn(0, [(0, 1, 2)])
            r1 = fn(3, [(0, 1, 3)])
            if r0 != 0 or not isinstance(r1, int) or r1 % 2 != 0:
                return False, "even_or_zero_invariant_violated"
            return True, "ok"

        return orc, mic

    o5, m5 = make_ch_dag_even()
    tasks.append(BenchmarkTask(tid5, "code_patch", "hard", prompt5, "", ("", "", ""), o5, m5))
    return tasks


# ===========================================================================
# 2. Domain: Scientific Kinetics, Speciation & Nonlinear ODEs (15 Tasks)
# ===========================================================================


def _build_live_scientific_tasks() -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []

    # --- 5 EASY SCIENTIFIC TASKS: Surface vs Bulk Kinetics Dimensional Analysis ---
    configs_easy = [
        (2, 1, 3),
        (1, 2, 2),  # 2D catalytic surface!
        (2, 2, 3),
        (3, 1, 2),  # 2D surface 4th order!
        (1, 1, 2),
    ]
    for i, (a_o, b_o, d_vol) in enumerate(configs_easy, start=1):
        tot = a_o + b_o
        u_vec = [d_vol * (tot - 1), 0, -1, 1 - tot]
        tid = f"live_sci_easy_{i:02d}"
        prompt = (
            f"Determine the overall reaction order and SI dimensional exponent vector `[L, M, T, mol]` "
            f"for rate constant `k` in `r = k [A]^{a_o} [B]^{b_o}` inside a {d_vol}D phase "
            f"(concentration `[X]` has units `mol * L^-{d_vol}`, reaction rate `r` has units `mol * L^-{d_vol} * s^-1`). "
            f'Output ONLY raw JSON `{{"order": int, "units": [L, M, T, mol]}}`.'
        )

        def make_se(t_ord: int, uv: List[int]) -> Tuple[Callable, Callable]:
            def orc(raw: str) -> bool:
                obj = extract_clean_json(raw)
                return obj.get("order") == t_ord and obj.get("units") == uv

            def mic(raw: str) -> Tuple[bool, str]:
                try:
                    obj = extract_clean_json(raw)
                except Exception as e:
                    return False, str(e)
                u = obj.get("units")
                if not (isinstance(u, list) and len(u) == 4 and all(isinstance(x, int) for x in u)):
                    return False, "invalid_units_schema"
                if u[1] != 0 or u[2] != -1:
                    return False, "molar_rate_M0_Tneg1_violated"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_se(tot, u_vec)
        tasks.append(BenchmarkTask(tid, "scientific_symbolic", "easy", prompt, "", ("", "", ""), o_fn, m_fn))

    # --- 5 MEDIUM SCIENTIFIC TASKS: 2x2 ODE Jacobian Spectrum & Classification ---
    mats = [
        (-3, 2, -5, -1),   # tr=-4, det=13, disc=16-52=-36 -> stable_spiral, abscissa=-2.0
        (-6, 2, 3, -5),    # tr=-11, det=24, disc=121-96=25 -> stable_node, eigs=(-3, -8), abscissa=-3.0
        (2, 5, 3, -4),     # tr=-2, det=-23 -> saddle, eigs=(-1 +/- sqrt(24)), abscissa=3.8990
        (1, -4, 3, 3),     # tr=4, det=15, disc=16-60=-44 -> unstable_spiral, abscissa=2.0
        (5, 3, 2, 6),      # tr=11, det=24, disc=121-96=25 -> unstable_node, eigs=(8, 3), abscissa=8.0
    ]
    for i, (a11, a12, a21, a22) in enumerate(mats, start=1):
        J = np.array([[a11, a12], [a21, a22]], dtype=float)
        tr = int(a11 + a22)
        det = int(a11 * a22 - a12 * a21)
        disc = tr * tr - 4 * det
        absc = round(float(np.max(np.real(np.linalg.eigvals(J)))), 4)
        if det < 0:
            cls = "saddle"
        elif disc < 0:
            cls = "stable_spiral" if tr < 0 else "unstable_spiral"
        else:
            cls = "stable_node" if tr < 0 else "unstable_node"
        tid = f"live_sci_med_{i:02d}"
        prompt = (
            f"For linear dynamical system `dx/dt = J x` with `J = [[{a11}, {a12}], [{a21}, {a22}]]`, "
            f"compute integer `trace`, integer `det`, `spectral_abscissa` (`max Re(lambda)` rounded to 4 decimals), "
            f"and `classification` in `{{stable_node, stable_spiral, saddle, unstable_node, unstable_spiral}}`. "
            f'Output ONLY raw JSON `{{"classification": str, "det": int, "spectral_abscissa": float, "trace": int}}`.'
        )

        def make_sm(tr_t: int, dt_t: int, ab_t: float, cl_t: str) -> Tuple[Callable, Callable]:
            valid_c = {"stable_node", "stable_spiral", "saddle", "unstable_node", "unstable_spiral"}

            def orc(raw: str) -> bool:
                obj = extract_clean_json(raw)
                return (
                    obj.get("trace") == tr_t
                    and obj.get("det") == dt_t
                    and obj.get("classification") == cl_t
                    and abs(float(obj.get("spectral_abscissa", 999.0)) - ab_t) <= 2e-3
                )

            def mic(raw: str) -> Tuple[bool, str]:
                try:
                    obj = extract_clean_json(raw)
                except Exception as e:
                    return False, str(e)
                c = obj.get("classification")
                d = obj.get("det")
                sa = obj.get("spectral_abscissa")
                if c not in valid_c or not isinstance(d, int) or not isinstance(sa, (int, float)):
                    return False, "schema_invalid"
                if c == "saddle" and not (d < 0 and sa > 0):
                    return False, "saddle_sign_mismatch"
                if c.startswith("stable") and not (d > 0 and sa < 0):
                    return False, "stable_sign_mismatch"
                if c.startswith("unstable") and not (d > 0 and sa > 0):
                    return False, "unstable_sign_mismatch"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_sm(tr, det, absc, cls)
        tasks.append(BenchmarkTask(tid, "scientific_symbolic", "medium", prompt, "", ("", "", ""), o_fn, m_fn))

    # --- 5 HARD SCIENTIFIC TASKS: Diprotic Speciation Kinetics (3) + Turing PDE Dispersion (2) ---
    spec_cases = [
        # (pKa1, pKa2, k0, k1, k2, pH)
        (4.20, 8.80, 12.0, 4.5e3, 1.8e6, 7.50),   # alpha1=0.9518, alpha2=0.0477 -> A2- dominates rate!
        (3.80, 9.40, 8.0, 2.2e3, 3.5e6, 7.60),    # 7.6-9.4=-1.8 (0.01585 * 3.5e6 = 55,475 vs 2200) -> A2- dominates!
        (4.50, 9.90, 15.0, 1.2e4, 2.0e5, 7.00),   # 7.0-9.9=-2.9 (0.00126 * 2e5 = 251 vs 11,947) -> HA- dominates!
    ]
    for idx, (pka1, pka2, k0, k1, k2, ph) in enumerate(spec_cases, start=1):
        h = 10.0 ** (-ph)
        ka1 = 10.0 ** (-pka1)
        ka2 = 10.0 ** (-pka2)
        denom = h * h + ka1 * h + ka1 * ka2
        c0 = (h * h / denom) * k0
        c1 = (ka1 * h / denom) * k1
        c2 = (ka1 * ka2 / denom) * k2
        k_app_true = c0 + c1 + c2
        dom_true = [("H2A", c0), ("HA-", c1), ("A2-", c2)]
        dom_species = max(dom_true, key=lambda x: x[1])[0]

        tid = f"live_sci_hard_{idx:02d}"
        prompt = (
            f"Solve this diprotic aqueous ozonation speciation kinetics problem and output ONLY raw JSON "
            f'`{{"k_app": float, "dominant_species": str}}` (`k_app` rounded to 2 decimals; '
            f'`dominant_species` in `{{"H2A", "HA-", "A2-"}}`).\n'
            f"Parameters: `pKa1 = {pka1:.2f}`, `pKa2 = {pka2:.2f}`, intrinsic second-order ozone rate constants "
            f"`k_H2A = {k0}`, `k_HA = {k1}`, `k_A2 = {k2}` M^-1 s^-1 at `pH = {ph:.2f}`.\n"
            f"Compute exact `k_app = alpha_0 * k_H2A + alpha_1 * k_HA + alpha_2 * k_A2` AND identify "
            f"which species contributes the LARGEST share of overall reaction rate `alpha_i * k_i`."
        )

        def make_sh_spec(kt: float, ds: str, k_max: float, k_min: float) -> Tuple[Callable, Callable]:
            def orc(raw: str) -> bool:
                obj = extract_clean_json(raw)
                kv = float(obj.get("k_app", -1.0))
                # Allow 0.5% relative numerical rounding tolerance on 10^-pH
                return obj.get("dominant_species") == ds and abs(kv - kt) <= max(1.0, 0.005 * kt)

            def mic(raw: str) -> Tuple[bool, str]:
                try:
                    obj = extract_clean_json(raw)
                except Exception as e:
                    return False, str(e)
                kv = obj.get("k_app")
                sp = obj.get("dominant_species")
                if sp not in ("H2A", "HA-", "A2-") or not isinstance(kv, (int, float)):
                    return False, "schema_invalid"
                # Physical convex-combination bound: min(k_i) <= k_app <= max(k_i)
                if not (k_min <= float(kv) <= k_max):
                    return False, "convex_hull_rate_bound_violated"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_sh_spec(k_app_true, dom_species, max(k0, k1, k2), min(k0, k1, k2))
        tasks.append(BenchmarkTask(tid, "scientific_symbolic", "hard", prompt, "", ("", "", ""), o_fn, m_fn))

    # Hard 4 & 5: Schnakenberg Reaction-Diffusion PDE Turing Dispersion
    turing_cases = [
        (0.20, 0.95, 1.0, 18.0, 0.75),
        (0.35, 1.15, 1.0, 10.5, 1.05),
    ]
    for offset_idx, (ap, bp, du, dv, kw) in enumerate(turing_cases, start=4):
        us = ap + bp
        j11 = (bp - ap) / us
        j12 = us * us
        j21 = -2.0 * bp / us
        j22 = -(us * us)
        M_k = np.array([[j11 - du * kw * kw, j12], [j21, j22 - dv * kw * kw]], dtype=float)
        da_true = round(float(np.max(np.real(np.linalg.eigvals(M_k)))), 4)
        frob_true = round(float(np.sum(M_k * M_k)), 4)
        reg_true = "turing_unstable" if da_true > 0 else "dispersion_stable"
        tid = f"live_sci_hard_{offset_idx:02d}"
        prompt = (
            f"For Schnakenberg reaction-diffusion PDE `du/dt = a - u + u^2 v + d_u nabla^2 u`, "
            f"`dv/dt = b - u^2 v + d_v nabla^2 v` with `a={ap:.2f}`, `b={bp:.2f}`, `d_u={du:.1f}`, "
            f"`d_v={dv:.1f}` at Fourier wavenumber `k={kw:.2f}`:\n"
            f"1. Find homogeneous equilibrium `(u*, v*)` and Jacobian `J(u*, v*)`.\n"
            f"2. Form dispersion matrix `M(k) = J(u*, v*) - k^2 diag(d_u, d_v)`.\n"
            f"3. Return ONLY raw JSON `{{\"dispersion_abscissa\": float, \"frob_norm_sq\": float, \"regime\": str, \"u_star\": float}}` "
            f"(4 decimals; `regime` in `{{\"turing_unstable\", \"dispersion_stable\"}}`, unstable iff `dispersion_abscissa > 0`)."
        )

        def make_sh_tur(ut: float, ft: float, at: float, rt: str) -> Tuple[Callable, Callable]:
            def orc(raw: str) -> bool:
                obj = extract_clean_json(raw)
                return (
                    obj.get("regime") == rt
                    and abs(float(obj.get("u_star", -99)) - ut) <= 2e-3
                    and abs(float(obj.get("frob_norm_sq", -99)) - ft) <= 2e-2
                    and abs(float(obj.get("dispersion_abscissa", -99)) - at) <= 3e-3
                )

            def mic(raw: str) -> Tuple[bool, str]:
                try:
                    obj = extract_clean_json(raw)
                except Exception as e:
                    return False, str(e)
                reg = obj.get("regime")
                da = obj.get("dispersion_abscissa")
                fs = obj.get("frob_norm_sq")
                if reg not in ("turing_unstable", "dispersion_stable") or not isinstance(da, (int, float)) or not isinstance(fs, (int, float)):
                    return False, "schema_invalid"
                if fs <= 0 or (da > 0 and reg != "turing_unstable") or (da <= 0 and reg != "dispersion_stable"):
                    return False, "sign_or_norm_invariant_failed"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_sh_tur(round(us, 4), frob_true, da_true, reg_true)
        tasks.append(BenchmarkTask(tid, "scientific_symbolic", "hard", prompt, "", ("", "", ""), o_fn, m_fn))

    return tasks


# ===========================================================================
# 3. Domain: Multi-Constraint Evidence Cross-Checking (15 Tasks: 5E / 5M / 5H)
# ===========================================================================


def _build_live_evidence_tasks() -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []

    # --- 5 EASY EVIDENCE TASKS ---
    for i in range(1, 6):
        v_fresh = 140 + 12 * i
        v_stale = v_fresh + 25
        blk = i % 3
        sched = sorted(
            [s0, s1, s2]
            for s0 in range(3)
            for s1 in range(3)
            for s2 in range(3)
            if len({s0, s1, s2}) == 3 and s0 != blk and s1 < s2
        )[0]
        tid = f"live_evid_easy_{i:02d}"
        prompt = (
            f"Reconcile `metric_mw` (latest timestamp `ts` wins) and assign `[J0, J1, J2]` to distinct slots `0, 1, 2`:\n"
            f"  | Replica | ts=1120 | metric_mw={v_stale} |\n"
            f"  | Primary | ts=1185 | metric_mw={v_fresh} |\n"
            f"Constraints: `slot(J0) != {blk}`, `slot(J1) < slot(J2)`, lexicographically smallest `[s0, s1, s2]`.\n"
            f'Output ONLY raw JSON `{{"metric_mw": int, "schedule": [s0, s1, s2]}}`.'
        )

        def make_ee(vt: int, st: List[int], b0: int) -> Tuple[Callable, Callable]:
            def orc(raw: str) -> bool:
                obj = extract_clean_json(raw)
                return obj.get("metric_mw") == vt and obj.get("schedule") == st

            def mic(raw: str) -> Tuple[bool, str]:
                try:
                    obj = extract_clean_json(raw)
                except Exception as e:
                    return False, str(e)
                sc = obj.get("schedule")
                if not (isinstance(sc, list) and sorted(sc) == [0, 1, 2] and sc[0] != b0 and sc[1] < sc[2]):
                    return False, "schedule_constraint_failed"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_ee(v_fresh, sched, blk)
        tasks.append(BenchmarkTask(tid, "evidence_crosscheck", "easy", prompt, "", ("", "", ""), o_fn, m_fn))

    # --- 5 MEDIUM EVIDENCE TASKS (Zero hints—model must identify corrupted sources itself!) ---
    for i in range(1, 6):
        base = 300 + 15 * i
        forbid = i % 4
        sched4 = sorted(
            [s0, s1, s2, s3]
            for s0 in range(4)
            for s1 in range(4)
            for s2 in range(4)
            for s3 in range(4)
            if len({s0, s1, s2, s3}) == 4 and s0 < s2 and s1 < s3 and s0 != forbid
        )[0]
        tid = f"live_evid_med_{i:02d}"
        prompt = (
            f"Reconcile `net_kWh` across 4 telemetry sources and schedule `[T0, T1, T2, T3]` onto permutation `0..3`:\n"
            f"  | S_Alpha | tier=1 | ts=1950 | crc=FAIL | net_kWh={base + 110} |\n"
            f"  | S_Beta  | tier=2 | ts=1810 | crc=PASS | net_kWh={base - 30} |\n"
            f"  | S_Gamma | tier=2 | ts=1910 | crc=PASS | net_kWh={base + 10} |\n"
            f"  | S_Delta | tier=1 | ts=1750 | crc=FAIL | net_kWh={base + 80} |\n"
            f"Rules: Discard all `crc=FAIL` rows (report their names sorted in `invalid_sources`). Among `crc=PASS` rows, "
            f"lowest tier number wins (`tier=1` highest priority), tie-break by largest `ts`. "
            f"Schedule rules: `slot(T0) < slot(T2)`, `slot(T1) < slot(T3)`, `slot(T0) != {forbid}`, lexicographically smallest.\n"
            f'Output ONLY raw JSON `{{"invalid_sources": [str, ...], "net_kWh": int, "schedule": [int, int, int, int]}}`.'
        )

        def make_em(vt: int, sc4: List[int], fb: int) -> Tuple[Callable, Callable]:
            def orc(raw: str) -> bool:
                obj = extract_clean_json(raw)
                return (
                    obj.get("net_kWh") == vt
                    and obj.get("schedule") == sc4
                    and obj.get("invalid_sources") == ["S_Alpha", "S_Delta"]
                )

            def mic(raw: str) -> Tuple[bool, str]:
                try:
                    obj = extract_clean_json(raw)
                except Exception as e:
                    return False, str(e)
                sc = obj.get("schedule")
                inv = obj.get("invalid_sources")
                if not (isinstance(sc, list) and sorted(sc) == [0, 1, 2, 3] and isinstance(inv, list)):
                    return False, "schema_invalid"
                if not (sc[0] < sc[2] and sc[1] < sc[3] and sc[0] != fb):
                    return False, "dag_constraint_failed"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_em(base + 10, sched4, forbid)
        tasks.append(BenchmarkTask(tid, "evidence_crosscheck", "medium", prompt, "", ("", "", ""), o_fn, m_fn))

    # --- 5 HARD EVIDENCE TASKS (Zero hints! Digit-sum parity + MW->kW conversion + Cross-sensor spoofing + 5-job DAG!) ---
    hard_evid_configs = [
        # (main_t1_old_mw, main_t1_spoof_kw, main_t2_ref_kw, aux_t1_badparity_kw, aux_t2_ok_kw)
        # Ensure digit sum of main_t1_old_mw*1000 is EVEN, main_t1_spoof_kw digit sum EVEN but |spoof - ref| > 200,
        # main_t2_ref_kw digit sum EVEN, aux_t1_badparity_kw digit sum ODD, aux_t2_ok_kw digit sum EVEN!
        ("4.20 MW", 4350, 4080, 315, 260, 0),  # 4200(sum=6 even, |4200-4080|=120<=200) + 260(sum=8 even) = 4460; corrupt: R2(spoof), R4(3+1+5=9 odd), R6(flag False)
        ("3.50 MW", 3680, 3420, 241, 170, 1),  # 3500(sum=8 even, |3500-3420|=80<=200) + 170(sum=8 even) = 3670; 3680(sum=17 odd AND >200), 241(sum=7 odd)
        ("5.10 MW", 5410, 5050, 421, 350, 0),  # 5100(sum=6 even, |5100-5050|=50<=200) + 350(sum=8 even) = 5450; 5410(sum=10 even, |5410-5050|=360>200!), 421(sum=7 odd)
        ("2.60 MW", 2910, 2530, 133, 220, 1),  # 2600(sum=8 even, |2600-2530|=70<=200) + 220(sum=4 even) = 2820; 2910(sum=12 even, |2910-2530|=380>200!), 133(sum=7 odd)
        ("6.20 MW", 6510, 6120, 511, 330, 0),  # 6200(sum=8 even, |6200-6120|=80<=200) + 330(sum=6 even) = 6530; 6510(sum=12 even, |6510-6120|=390>200!), 511(sum=7 odd)
    ]
    for idx, (m_mw_str, m_spoof, m_ref, a_odd, a_ok, ban0) in enumerate(hard_evid_configs, start=1):
        m_valid_kw = int(round(float(m_mw_str.split()[0]) * 1000))
        # Programmatically apply exact Rules 1, 2, and 3 on R1..R6 so ground truth is 100% consistent
        raw_rows = [
            ("R1", "Main", 1, 1920, m_valid_kw, True),
            ("R2", "Main", 1, 1950, m_spoof, True),
            ("R3", "Main", 2, 1950, m_ref, True),
            ("R4", "Aux", 1, 1940, a_odd, True),
            ("R5", "Aux", 2, 1950, a_ok, True),
            ("R6", "Aux", 3, 1960, 981, False),
        ]
        parity_corrupt = {
            rid
            for rid, _bus, _tier, _ts, kw_val, p_ok in raw_rows
            if (not p_ok) or (sum(int(ch) for ch in str(abs(kw_val))) % 2 == 1)
        }
        # Latest parity-valid tier=2 reference per bus
        ref_t2: Dict[str, int] = {}
        for rid, bus, tier, ts, kw_val, _p_ok in sorted(raw_rows, key=lambda x: x[3]):
            if tier == 2 and rid not in parity_corrupt:
                ref_t2[bus] = kw_val

        corrupt_set = set(parity_corrupt)
        for rid, bus, tier, _ts, kw_val, _p_ok in raw_rows:
            if tier == 1 and bus in ref_t2 and abs(kw_val - ref_t2[bus]) > 200:
                corrupt_set.add(rid)

        corrupt_true = sorted(corrupt_set)
        valid_main = [r for r in raw_rows if r[1] == "Main" and r[0] not in corrupt_set]
        valid_aux = [r for r in raw_rows if r[1] == "Aux" and r[0] not in corrupt_set]
        best_main = sorted(valid_main, key=lambda r: (r[2], -r[3]))[0][4]
        best_aux = sorted(valid_aux, key=lambda r: (r[2], -r[3]))[0][4]
        rec_true = best_main + best_aux

        sched5 = sorted(
            [s0, s1, s2, s3, s4]
            for s0 in range(5)
            for s1 in range(5)
            for s2 in range(5)
            for s3 in range(5)
            for s4 in range(5)
            if len({s0, s1, s2, s3, s4}) == 5
            and s0 < s2 < s4
            and s1 < s3
            and abs(s0 - s1) >= 2
            and s0 != ban0
        )[0]
        tid = f"live_evid_hard_{idx:02d}"
        prompt = (
            f"Audit the 6 telemetry rows below, normalize all loads to integer `kW` (`1 MW = 1000 kW`), "
            f"and schedule `[W0, W1, W2, W3, W4]` onto permutation `0..4`:\n"
            f"  | R1 | bus=Main | tier=1 | ts=1920 | load={m_mw_str} | parity_ok=True  |\n"
            f"  | R2 | bus=Main | tier=1 | ts=1950 | load={m_spoof} kW | parity_ok=True  |\n"
            f"  | R3 | bus=Main | tier=2 | ts=1950 | load={m_ref} kW | parity_ok=True  |\n"
            f"  | R4 | bus=Aux  | tier=1 | ts=1940 | load={a_odd} kW  | parity_ok=True  |\n"
            f"  | R5 | bus=Aux  | tier=2 | ts=1950 | load={a_ok} kW  | parity_ok=True  |\n"
            f"  | R6 | bus=Aux  | tier=3 | ts=1960 | load=981 kW  | parity_ok=False |\n"
            f"Audit & Reconciliation Rules (NO hints—verify every row yourself!):\n"
            f"1. Parity Rule: A row is CORRUPTED if `parity_ok=False` OR if the sum of base-10 digits of its integer kW value is ODD.\n"
            f"2. Anti-Spoofing Rule: Any `tier=1` row is ALSO CORRUPTED if its kW value differs by STRICTLY MORE than `200 kW` "
            f"from the latest parity-valid `tier=2` reference on the SAME bus.\n"
            f"3. `reconciled_kW` = valid highest-priority (`tier=1` > `tier=2` > `tier=3`, tie-break latest `ts`) `Main` kW + valid highest-priority `Aux` kW.\n"
            f"4. Schedule `[W0..W4]` onto `0..4`: `W0 < W2 < W4`, `W1 < W3`, `abs(W0 - W1) >= 2`, `W0 != {ban0}`, lexicographically smallest.\n"
            f'Output ONLY raw JSON `{{"corrupted_ids": [str, ...], "reconciled_kW": int, "schedule": [int, int, int, int, int]}}` (`corrupted_ids` sorted alphabetically).'
        )

        def make_eh(rt: int, c_ids: List[str], sc5: List[int], bw: int) -> Tuple[Callable, Callable]:
            def orc(raw: str) -> bool:
                obj = extract_clean_json(raw)
                return (
                    obj.get("reconciled_kW") == rt
                    and obj.get("corrupted_ids") == c_ids
                    and obj.get("schedule") == sc5
                )

            def mic(raw: str) -> Tuple[bool, str]:
                try:
                    obj = extract_clean_json(raw)
                except Exception as e:
                    return False, str(e)
                sc = obj.get("schedule")
                cid = obj.get("corrupted_ids")
                rk = obj.get("reconciled_kW")
                if not (isinstance(sc, list) and sorted(sc) == [0, 1, 2, 3, 4] and isinstance(cid, list) and isinstance(rk, int)):
                    return False, "schema_invalid"
                s0, s1, s2, s3, s4 = sc
                if not (s0 < s2 < s4 and s1 < s3 and abs(s0 - s1) >= 2 and s0 != bw):
                    return False, "dag_or_mutex_violated"
                if "R6" not in cid:
                    return False, "explicit_parity_false_missed"
                return True, "ok"

            return orc, mic

        o_fn, m_fn = make_eh(rec_true, corrupt_true, sched5, ban0)
        tasks.append(BenchmarkTask(tid, "evidence_crosscheck", "hard", prompt, "", ("", "", ""), o_fn, m_fn))

    return tasks


def build_live_evaluation_suite() -> List[BenchmarkTask]:
    """Build the complete 45-task (15 Easy / 15 Medium / 15 Hard) Live-LLM suite."""
    suite: List[BenchmarkTask] = []
    suite.extend(_build_live_code_tasks())
    suite.extend(_build_live_scientific_tasks())
    suite.extend(_build_live_evidence_tasks())
    return suite
