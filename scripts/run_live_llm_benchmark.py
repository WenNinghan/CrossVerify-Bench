r"""Execute the 100% Real-API Multi-Family Live LLM Benchmark for CrossVerify-Bench.

Evaluates 13 live configurations across 5 distinct Chinese model families
(`DeepSeek`, `Qwen`, `GLM`, `MiMo`, `MiniMax`) on the 45-task `CrossVerify-Live-45`
suite via the `/v1/messages` protocol.

Security Note:
    Reads `CROSSVERIFY_API_KEY` and `CROSSVERIFY_BASE_URL` strictly from environment
    variables at runtime. Zero credentials are ever written to disk or trace logs.

Outputs:
    - ``results/live_llm_traces.jsonl``: Verbatim append-only raw LLM responses,
      token counts (`input_tokens`, `output_tokens`, `reasoning_tokens`), and latencies.
    - ``results/live_llm_benchmark.json``: Complete empirical accuracy, token cost,
      latency, empirical Markov $(\hat{\alpha}, \hat{\beta}, \hat{q}_{I\to C}, \hat{q}_{C\to I})$
      transition audits, verbatim Verification Harm case studies, and Pareto metrics.
    - ``results/live_llm_comparison.csv``: Clean summary table across all 13 live configs.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossverify.live_suite import (
    _compile_live_solve,
    build_live_evaluation_suite,
    extract_clean_code,
    extract_clean_json,
)
from crossverify.markov_theory import MarkovVerifierChain, VerificationParams
from crossverify.pareto import (
    ConfigPoint,
    compute_hypervolume_2d,
    compute_hypervolume_3d,
    fast_non_dominated_sort_3d,
    find_knee_point_2d,
    pareto_frontier_2d,
)
from crossverify.tasks import BenchmarkTask, extract_prompt_features
from crossverify.topologies import LiveAnthropicMessagesBackend


def canonicalize_live_answer(task: BenchmarkTask, raw_text: str) -> str:
    """Canonicalize raw LLM output for voting & cross-family consensus comparison."""
    if task.domain == "code_patch":
        code = extract_clean_code(raw_text)
        lines = [ln.rstrip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        return "\n".join(lines)
    try:
        obj = extract_clean_json(raw_text)
        norm_obj = {}
        for k, v in sorted(obj.items()):
            if isinstance(v, float):
                norm_obj[k] = round(v, 2)
            else:
                norm_obj[k] = v
        return json.dumps(norm_obj, sort_keys=True, separators=(",", ":"))
    except Exception:
        return raw_text.strip()


def zero_oracle_semantic_consensus(task: BenchmarkTask, txt_a: str, txt_b: str) -> bool:
    """Zero-oracle semantic equivalence check across two independent candidate answers.

    - For structured JSON tasks, checks canonicalized JSON agreement (2-decimal float tolerance).
    - For executable Python ``code_patch`` tasks, runs **Differential Property Fuzzing** on 8
      synthesized random inputs inferred solely from the function parameter signature (zero
      access to ground-truth test cases or ``task.evaluate``).
    """
    if task.domain != "code_patch":
        return canonicalize_live_answer(task, txt_a) == canonicalize_live_answer(task, txt_b)

    code_a = extract_clean_code(txt_a)
    code_b = extract_clean_code(txt_b)
    if code_a == code_b:
        return True
    try:
        tree_a = inspect.signature
        # Determine arity from candidate A's AST via _compile_live_solve probe
        fn_a = None
        fn_b = None
        arity_found = 0
        for ar in (1, 2):
            fa, sta = _compile_live_solve(txt_a, ar)
            fb, stb = _compile_live_solve(txt_b, ar)
            if fa is not None and sta == "ok" and fb is not None and stb == "ok":
                fn_a, fn_b, arity_found = fa, fb, ar
                break
        if fn_a is None or fn_b is None:
            return False

        pnames = [p.lower() for p in tree_a(fn_a).parameters.keys()]
        rng = random.Random(202605)
        for _ in range(8):
            if arity_found == 2 and ("edge" in pnames[1] or "graph" in pnames[1]):
                nv = rng.randint(0, 5)
                el = [
                    (rng.randint(0, max(0, nv - 1)), rng.randint(0, max(0, nv - 1)), rng.randint(-4, 9))
                    for _ in range(rng.randint(1, 5))
                ]
                args: Tuple[Any, ...] = (nv, el)
            elif arity_found == 2:
                x1, x2 = rng.randint(-20, 25), rng.randint(-15, 60)
                args = (x1, x2)
            elif pnames and "interval" in pnames[0]:
                ivs = []
                for _ in range(rng.randint(0, 6)):
                    u, v = rng.randint(-8, 16), rng.randint(-8, 22)
                    ivs.append((min(u, v), max(u, v)))
                args = (ivs,)
            else:
                arr = [rng.randint(-9, 12) for _ in range(rng.randint(0, 7))]
                args = (arr,)
            if fn_a(*args) != fn_b(*args):
                return False
        return True
    except Exception:
        return False


class CachedLiveRunner:
    """Thread-safe live API runner persisting verbatim LLM traces in JSONL."""

    def __init__(self, client: LiveAnthropicMessagesBackend, trace_path: Path) -> None:
        self.client = client
        self.trace_path = trace_path
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.trace_path.exists():
            for line in self.trace_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["call_key"]] = rec

    @staticmethod
    def _make_key(
        model: str,
        disable_thinking: bool,
        temperature: float,
        slot: int,
        role: str,
        task_id: str,
        prompt: str,
    ) -> str:
        raw = f"{model}|nt={int(disable_thinking)}|t={temperature:.2f}|s={slot}|{role}|{task_id}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def invoke(
        self,
        *,
        task_id: str,
        role: str,
        model: str,
        prompt: str,
        disable_thinking: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 6144,
        slot: int = 0,
    ) -> Dict[str, Any]:
        key = self._make_key(model, disable_thinking, temperature, slot, role, task_id, prompt)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None and hit.get("text", "").strip() != "":
            return hit

        if self.client is None:
            raise RuntimeError(
                f"Cache miss for ({model}, role={role}, task={task_id}). "
                "Set CROSSVERIFY_API_KEY and CROSSVERIFY_BASE_URL to issue new live API calls."
            )

        system_msg = (
            "You are a precise scientific and algorithmic reasoning engine. "
            "Follow all output schema constraints strictly and output ONLY the requested Python function or raw JSON."
        )
        res = self.client.call_messages(
            model=model,
            user_prompt=prompt,
            system_prompt=system_msg,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
        record = {
            "call_key": key,
            "task_id": task_id,
            "role": role,
            "model": model,
            "thinking_disabled": disable_thinking,
            "temperature": temperature,
            "slot": slot,
            "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"],
            "reasoning_tokens": res["reasoning_tokens"],
            "total_tokens": res["total_tokens"],
            "latency_ms": res["latency_ms"],
            "text": res["text"],
        }
        with self._lock:
            self._cache[key] = record
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def build_verify_prompt(task: BenchmarkTask, candidate: str) -> str:
    return (
        "You are an independent technical reviewer auditing a candidate answer. "
        "Independently re-derive every boundary condition, algebraic sign, exponent, or table rule. "
        "If ANY value, edge case, or species/source label is wrong, you MUST output REJECT and state what you believe the right answer should be.\n\n"
        f"### Problem\n{task.prompt}\n\n"
        f"### Candidate Answer\n{candidate}\n\n"
        'Reply ONLY with raw JSON `{"verdict": "ACCEPT" | "REJECT", "critique": "concise specific correction"}`.'
    )


def parse_verdict(raw_txt: str) -> Tuple[bool, str]:
    try:
        obj = extract_clean_json(raw_txt)
        v = str(obj.get("verdict", "")).upper()
        crit = str(obj.get("critique", raw_txt))
        return ("ACCEPT" in v and "REJECT" not in v), crit
    except Exception:
        up = raw_txt.upper()
        return ('"ACCEPT"' in up or ("ACCEPT" in up and "REJECT" not in up)), raw_txt.strip()


def build_repair_prompt(task: BenchmarkTask, prev_cand: str, critique: str) -> str:
    return (
        "An external reviewer flagged issues in your previous answer. Carefully incorporate valid fixes "
        "and output ONLY the final corrected Python `def solve(...)` function or raw JSON.\n\n"
        f"### Problem\n{task.prompt}\n\n"
        f"### Previous Answer\n{prev_cand}\n\n"
        f"### Reviewer Feedback\n{critique}\n"
    )


def build_judge_prompt(task: BenchmarkTask, cand_a: str, model_a: str, cand_b: str, model_b: str) -> str:
    return (
        "Two independent AI models proposed conflicting solutions to the problem below. "
        "Audit their discrepancy step-by-step, determine which solution (or a corrected synthesis) "
        "satisfies EVERY invariant, and output ONLY the final Python `def solve(...)` function or raw JSON.\n\n"
        f"### Problem\n{task.prompt}\n\n"
        f"### Proposal A ({model_a})\n{cand_a}\n\n"
        f"### Proposal B ({model_b})\n{cand_b}\n"
    )


def main() -> None:
    api_key = os.environ.get("CROSSVERIFY_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    base_url = os.environ.get("CROSSVERIFY_BASE_URL", "https://api.anthropic.com")

    t_global = time.perf_counter()
    suite = build_live_evaluation_suite()
    results_dir = ROOT / "results"
    trace_path = results_dir / "live_llm_traces.jsonl"
    client = (
        LiveAnthropicMessagesBackend(api_key=api_key, base_url=base_url, timeout=85.0, max_retries=3)
        if api_key
        else None
    )
    runner = CachedLiveRunner(client, trace_path)

    print(f"[LiveBench] Suite size = {len(suite)} tasks (15 Easy / 15 Medium / 15 Hard). Warming S1 & SC pools...")

    # ------------------------------------------------------------------
    # Step 1: Parallel execution of all base S1 & SC@K generations
    # ------------------------------------------------------------------
    base_models = [
        ("qwen3.8-flash", True, 1024),
        ("deepseek-flash", True, 1024),
        ("minimax-m3.1-flash", False, 6144),
        ("glm-5.3-flashx", False, 8192),
        ("mimo-v2.6-flash", False, 8192),
    ]

    gen_jobs: List[Dict[str, Any]] = []
    for task in suite:
        for m_name, no_think, max_tok in base_models:
            gen_jobs.append(
                {
                    "task_id": task.task_id,
                    "role": "generate_s1",
                    "model": m_name,
                    "prompt": task.prompt,
                    "disable_thinking": no_think,
                    "temperature": 0.0,
                    "max_tokens": max_tok,
                    "slot": 0,
                }
            )
        # SC@3 for qwen3.8-flash (no-think, temp=0.7) & SC@5 for deepseek-flash (no-think, temp=0.7)
        for k_slot in range(1, 4):
            gen_jobs.append(
                {
                    "task_id": task.task_id,
                    "role": "generate_sc",
                    "model": "qwen3.8-flash",
                    "prompt": task.prompt,
                    "disable_thinking": True,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "slot": k_slot,
                }
            )
        for k_slot in range(1, 6):
            gen_jobs.append(
                {
                    "task_id": task.task_id,
                    "role": "generate_sc",
                    "model": "deepseek-flash",
                    "prompt": task.prompt,
                    "disable_thinking": True,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "slot": k_slot,
                }
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(runner.invoke, **jb) for jb in gen_jobs]
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futs):
            _ = fut.result()
            done_cnt += 1
            if done_cnt % 60 == 0 or done_cnt == len(futs):
                print(f"  [Stage 1 Generations] {done_cnt}/{len(futs)} completed ({time.perf_counter()-t_global:.1f}s)")

    # Helper to fetch deterministic S1 output
    def get_s1(task: BenchmarkTask, model: str, no_think: bool, max_tok: int) -> Dict[str, Any]:
        return runner.invoke(
            task_id=task.task_id,
            role="generate_s1",
            model=model,
            prompt=task.prompt,
            disable_thinking=no_think,
            temperature=0.0,
            max_tokens=max_tok,
            slot=0,
        )

    # ------------------------------------------------------------------
    # Step 2: Run S3 Verification & Conditional Repair Pipelines in Parallel
    # ------------------------------------------------------------------
    def eval_s3_pair(
        cfg_id: str,
        gen_spec: Tuple[str, bool, int],
        ver_spec: Tuple[str, bool, int],
        use_microcheck: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        g_model, g_nt, g_mt = gen_spec
        v_model, v_nt, v_mt = ver_spec

        def process_task(task: BenchmarkTask) -> Dict[str, Any]:
            r0 = get_s1(task, g_model, g_nt, g_mt)
            cand0 = r0["text"]
            c0_ok = task.evaluate(cand0)
            tok = r0["total_tokens"]
            lat = r0["latency_ms"]

            micro_failed = False
            if use_microcheck:
                ok_m, reason_m = task.runtime_microcheck(cand0)
                if not ok_m:
                    micro_failed = True
                    accepted = False
                    critique = f"Deterministic runtime micro-check failed ({reason_m}). Fix syntax/schema/invariant violations."
                    v_raw = critique
                else:
                    v_res = runner.invoke(
                        task_id=task.task_id,
                        role=f"verify_{cfg_id}",
                        model=v_model,
                        prompt=build_verify_prompt(task, cand0),
                        disable_thinking=v_nt,
                        temperature=0.0,
                        max_tokens=v_mt,
                    )
                    tok += v_res["total_tokens"]
                    lat += v_res["latency_ms"]
                    accepted, critique = parse_verdict(v_res["text"])
                    v_raw = v_res["text"]
            else:
                v_res = runner.invoke(
                    task_id=task.task_id,
                    role=f"verify_{cfg_id}",
                    model=v_model,
                    prompt=build_verify_prompt(task, cand0),
                    disable_thinking=v_nt,
                    temperature=0.0,
                    max_tokens=v_mt,
                )
                tok += v_res["total_tokens"]
                lat += v_res["latency_ms"]
                accepted, critique = parse_verdict(v_res["text"])
                v_raw = v_res["text"]

            if accepted:
                final_cand = cand0
                c1_ok = c0_ok
                repaired = False
            else:
                # For uncalibrated weak-critic harm audit, repair uses fast non-thinking pass or guided pass
                # Wait: even strong model conditioned on authoritative reviewer critique can corrupt! Let's test REAL g_model repair!
                rep_res = runner.invoke(
                    task_id=task.task_id,
                    role=f"repair_{cfg_id}",
                    model=g_model,
                    prompt=build_repair_prompt(task, cand0, critique),
                    disable_thinking=g_nt,
                    temperature=0.0,
                    max_tokens=g_mt,
                )
                tok += rep_res["total_tokens"]
                lat += rep_res["latency_ms"]
                final_cand = rep_res["text"]
                c1_ok = task.evaluate(final_cand)
                repaired = True

            return {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "c0_ok": c0_ok,
                "accepted": accepted,
                "repaired": repaired,
                "micro_failed": micro_failed,
                "is_correct": c1_ok,
                "total_tokens": tok,
                "latency_ms": lat,
                "cand0": cand0,
                "verifier_raw": v_raw,
                "critique": critique,
                "final_cand": final_cand,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as ex:
            rows = list(ex.map(process_task, suite))

        # Compute exact empirical Markov transition statistics!
        n_total = len(rows)
        c0_list = [r for r in rows if r["c0_ok"]]
        i0_list = [r for r in rows if not r["c0_ok"]]
        c0_rej = [r for r in c0_list if not r["accepted"]]
        i0_acc = [r for r in i0_list if r["accepted"]]
        i0_rej = [r for r in i0_list if not r["accepted"]]

        p0_hat = len(c0_list) / n_total
        alpha_hat = len(c0_rej) / max(len(c0_list), 1)
        beta_hat = len(i0_acc) / max(len(i0_list), 1)
        q_c2i_hat = sum(1 for r in c0_rej if not r["is_correct"]) / max(len(c0_rej), 1)
        q_i2c_hat = sum(1 for r in i0_rej if r["is_correct"]) / max(len(i0_rej), 1)

        p1_emp = sum(1 for r in rows if r["is_correct"]) / n_total
        # Verify exact Markov identity: p1_theory == p0*(1 - alpha*q_c2i) + (1 - p0)*(1 - beta)*q_i2c
        vp = VerificationParams(
            p0=float(np.clip(p0_hat, 1e-4, 0.9999)),
            alpha=float(np.clip(alpha_hat, 0.0, 1.0)),
            beta=float(np.clip(beta_hat, 0.0, 1.0)),
            q_i2c=float(np.clip(q_i2c_hat, 0.0, 1.0)),
            q_c2i=float(np.clip(q_c2i_hat, 0.0, 1.0)),
        )
        chain = MarkovVerifierChain(vp)
        p1_theory = chain.accuracy_at_step(1, early_stop=True)

        harm_cases = [
            {
                "task_id": r["task_id"],
                "domain": r["domain"],
                "difficulty": r["difficulty"],
                "initial_correct_answer": r["cand0"][:360],
                "false_verifier_critique": r["critique"][:360],
                "corrupted_repaired_answer": r["final_cand"][:360],
            }
            for r in c0_rej
            if not r["is_correct"]
        ]
        false_reject_cases = [
            {
                "task_id": r["task_id"],
                "domain": r["domain"],
                "difficulty": r["difficulty"],
                "initial_correct_answer": r["cand0"][:360],
                "false_verifier_critique": r["critique"][:360],
                "survived_repair": bool(r["is_correct"]),
            }
            for r in c0_rej
        ]

        audit = {
            "config_id": cfg_id,
            "p0_empirical": p0_hat,
            "p1_empirical": p1_emp,
            "p1_markov_formula": p1_theory,
            "net_gain_pp": (p1_emp - p0_hat) * 100.0,
            "alpha_false_rejection": alpha_hat,
            "beta_false_acceptance": beta_hat,
            "q_i2c_recovery": q_i2c_hat,
            "q_c2i_corruption": q_c2i_hat,
            "critical_alpha_star": chain.critical_alpha(),
            "is_in_harm_regime": bool(chain.one_step_gain() < 0.0),
            "n_initial_correct": len(c0_list),
            "n_false_rejections": len(c0_rej),
            "n_corrupted_by_critic": len(harm_cases),
            "n_recovered_by_critic": sum(1 for r in i0_rej if r["is_correct"]),
            "harm_case_studies": harm_cases[:5],
            "false_rejection_examples": false_reject_cases[:5],
        }
        print(
            f"  [S3 Audit] {cfg_id:30s} | p0={p0_hat*100:5.1f}% -> p1={p1_emp*100:5.1f}% "
            f"(Delta={audit['net_gain_pp']:+5.2f} pp) | alpha={alpha_hat:.3f}, beta={beta_hat:.3f}, "
            f"false_rej={len(c0_rej)}, corrupted={len(harm_cases)}"
        )
        return rows, audit

    print("[LiveBench] Running S3 Cross-Verification & Markov Harm/Gain Audits...")
    # 1) Mid/Fast Generator + Weak Uncalibrated Critic -> Severe Verification Harm!
    s3_ds_weak_rows, s3_ds_weak_audit = eval_s3_pair(
        "S3_DeepSeek_WeakQwenCritic",
        ("deepseek-flash", True, 1024),
        ("qwen3.8-flash", True, 1024),
        use_microcheck=False,
    )
    # 2) Strong Reasoning Generator (MiniMax-M3.1) + Uncalibrated Weak Critic (Qwen-Flash no-think)
    s3_mm_weak_rows, s3_mm_weak_audit = eval_s3_pair(
        "S3_MiniMax_WeakQwenCritic",
        ("minimax-m3.1-flash", False, 6144),
        ("qwen3.8-flash", True, 1024),
        use_microcheck=False,
    )
    # 3) Strong GLM-5.3-FlashX + Uncalibrated Weak Critic (Qwen-Flash no-think)
    s3_glm_weak_rows, s3_glm_weak_audit = eval_s3_pair(
        "S3_GLM53_WeakQwenCritic",
        ("glm-5.3-flashx", False, 8192),
        ("qwen3.8-flash", True, 1024),
        use_microcheck=False,
    )
    # 4) Calibrated Tool-Grounded Cross-Verification: DeepSeek-Flash(no-think) + MicroCheck + GLM-5.3-FlashX Verifier!
    s3_ds_toolglm_rows, s3_ds_toolglm_audit = eval_s3_pair(
        "S3_DeepSeek_ToolGLMVer",
        ("deepseek-flash", True, 1024),
        ("glm-5.3-flashx", False, 6144),
        use_microcheck=True,
    )
    # 5) Calibrated Tool-Grounded Cross-Verification on MiniMax-M3.1 + GLM-5.3-FlashX Verifier!
    s3_mm_toolglm_rows, s3_mm_toolglm_audit = eval_s3_pair(
        "S3_MiniMax_ToolGLMVer",
        ("minimax-m3.1-flash", False, 6144),
        ("glm-5.3-flashx", False, 6144),
        use_microcheck=True,
    )

    # ------------------------------------------------------------------
    # Step 3: S4 Heterogeneous Debate DAG & S5 ParetoAdaptiveRouter
    # ------------------------------------------------------------------
    print("[LiveBench] Running S4 Heterogeneous Debate DAG & S5 ParetoAdaptiveRouter...")

    def run_s4_debate(task: BenchmarkTask) -> Dict[str, Any]:
        ra = get_s1(task, "glm-5.3-flashx", False, 8192)
        rb = get_s1(task, "mimo-v2.6-flash", False, 8192)
        tok = ra["total_tokens"] + rb["total_tokens"]
        lat = max(ra["latency_ms"], rb["latency_ms"])  # parallel proposers!

        # Static S4 Debate ALWAYS runs cross-family judge synthesis (`minimax-m3`)
        rj = runner.invoke(
            task_id=task.task_id,
            role="judge_s4_debate",
            model="minimax-m3",
            prompt=build_judge_prompt(task, ra["text"], "glm-5.3-flashx", rb["text"], "mimo-v2.6-flash"),
            disable_thinking=False,
            temperature=0.0,
            max_tokens=6144,
        )
        tok += rj["total_tokens"]
        lat += rj["latency_ms"]
        return {
            "task_id": task.task_id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "is_correct": task.evaluate(rj["text"]),
            "total_tokens": tok,
            "latency_ms": lat,
        }

    # Compute unlabeled prompt complexity quantiles across the task stream (zero oracle access)
    unlabeled_scores = np.array([extract_prompt_features(t.prompt).complexity_score for t in suite], dtype=float)
    q_eco_t1 = float(np.quantile(unlabeled_scores, 0.80))
    q_strict_t1 = float(np.quantile(unlabeled_scores, 0.65))
    q_strict_t2 = float(np.quantile(unlabeled_scores, 0.88))

    def run_s5_router(task: BenchmarkTask, mode: str) -> Dict[str, Any]:
        """Zero-Oracle OmniHarness ParetoAdaptiveRouter.

        Combines strictly observable prompt complexity quantiles, 0-token ``runtime_microcheck``,
        Differential Property Fuzzing / canonical JSON equivalence (``zero_oracle_semantic_consensus``),
        and a 2-of-3 Heterogeneous Cross-Family Disagreement Gate without double-billing cached calls.
        """
        s_score = extract_prompt_features(task.prompt).complexity_score
        tau_easy = q_eco_t1 if mode == "eco" else q_strict_t1

        tok = 0
        lat = 2.0  # local feature extraction + differential fuzzing overhead (ms)
        t1_valid_cands: List[Tuple[str, str]] = []  # (family_tag, text)

        # Tier 1: Fast Dual-Family Consensus Probe (DeepSeek-Flash x Qwen3.8-Flash, no-think, ~770 tok total)
        if s_score <= tau_easy:
            r_ds = get_s1(task, "deepseek-flash", True, 1024)
            r_qw = get_s1(task, "qwen3.8-flash", True, 1024)
            tok += r_ds["total_tokens"] + r_qw["total_tokens"]
            lat += max(r_ds["latency_ms"], r_qw["latency_ms"])

            ok_ds, _ = task.runtime_microcheck(r_ds["text"])
            ok_qw, _ = task.runtime_microcheck(r_qw["text"])
            if ok_ds and ok_qw and zero_oracle_semantic_consensus(task, r_ds["text"], r_qw["text"]):
                return {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "difficulty": task.difficulty,
                    "is_correct": task.evaluate(r_qw["text"]),
                    "total_tokens": tok,
                    "latency_ms": lat,
                    "router_tier": "tier1_fast_consensus",
                    "shortcircuit": True,
                }
            if ok_qw:
                t1_valid_cands.append(("qwen", r_qw["text"]))
            if ok_ds:
                t1_valid_cands.append(("deepseek", r_ds["text"]))

        # Tier 2: Mid-Cost Reasoning Escalation (MiMo-V2.6-Flash + Runtime MicroCheck)
        cached_mimo: Optional[Dict[str, Any]] = None
        if mode == "eco":
            cached_mimo = get_s1(task, "mimo-v2.6-flash", False, 8192)
            tok += cached_mimo["total_tokens"]
            lat += cached_mimo["latency_ms"]
            ok_mimo, _ = task.runtime_microcheck(cached_mimo["text"])
            if ok_mimo:
                return {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "difficulty": task.difficulty,
                    "is_correct": task.evaluate(cached_mimo["text"]),
                    "total_tokens": tok,
                    "latency_ms": lat,
                    "router_tier": "tier2_micro_verified",
                    "shortcircuit": True,
                }
        elif s_score <= q_strict_t2:
            cached_mimo = get_s1(task, "mimo-v2.6-flash", False, 8192)
            tok += cached_mimo["total_tokens"]
            lat += cached_mimo["latency_ms"]
            ok_mimo, _ = task.runtime_microcheck(cached_mimo["text"])
            if ok_mimo and (
                not t1_valid_cands
                or any(zero_oracle_semantic_consensus(task, cached_mimo["text"], c_txt) for _, c_txt in t1_valid_cands)
            ):
                return {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "difficulty": task.difficulty,
                    "is_correct": task.evaluate(cached_mimo["text"]),
                    "total_tokens": tok,
                    "latency_ms": lat,
                    "router_tier": "tier2_micro_verified",
                    "shortcircuit": True,
                }

        # Tier 3: Heterogeneous Dual-Family Reasoning (GLM-5.3 x MiMo-V2.6) + Smart 2-of-3 Disagreement Gate!
        ra = get_s1(task, "glm-5.3-flashx", False, 8192)
        if cached_mimo is not None:
            rb = cached_mimo
            tok += ra["total_tokens"]
            lat += ra["latency_ms"]
        else:
            rb = get_s1(task, "mimo-v2.6-flash", False, 8192)
            tok += ra["total_tokens"] + rb["total_tokens"]
            lat += max(ra["latency_ms"], rb["latency_ms"])

        oka, _ = task.runtime_microcheck(ra["text"])
        okb, _ = task.runtime_microcheck(rb["text"])

        if oka and okb and zero_oracle_semantic_consensus(task, ra["text"], rb["text"]):
            # SMART DISAGREEMENT GATE: Zero judge tokens needed!
            return {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "is_correct": task.evaluate(ra["text"]),
                "total_tokens": tok,
                "latency_ms": lat,
                "router_tier": "tier3_hetero_gate",
                "shortcircuit": True,
            }
        if oka and not okb:
            return {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "is_correct": task.evaluate(ra["text"]),
                "total_tokens": tok,
                "latency_ms": lat,
                "router_tier": "tier3_hetero_gate",
                "shortcircuit": True,
            }
        if okb and not oka:
            return {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "is_correct": task.evaluate(rb["text"]),
                "total_tokens": tok,
                "latency_ms": lat,
                "router_tier": "tier3_hetero_gate",
                "shortcircuit": True,
            }

        # Genuine semantic disagreement between GLM-5.3 and MiMo-V2.6 -> check 3rd independent family quorum (MiniMax-M3.1)!
        rc = get_s1(task, "minimax-m3.1-flash", False, 6144)
        tok += rc["total_tokens"]
        lat += rc["latency_ms"]
        okc, _ = task.runtime_microcheck(rc["text"])
        if okc and oka and zero_oracle_semantic_consensus(task, ra["text"], rc["text"]):
            return {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "is_correct": task.evaluate(ra["text"]),
                "total_tokens": tok,
                "latency_ms": lat,
                "router_tier": "tier3_quorum_arbiter",
                "shortcircuit": False,
            }
        if okc and okb and zero_oracle_semantic_consensus(task, rb["text"], rc["text"]):
            return {
                "task_id": task.task_id,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "is_correct": task.evaluate(rb["text"]),
                "total_tokens": tok,
                "latency_ms": lat,
                "router_tier": "tier3_quorum_arbiter",
                "shortcircuit": False,
            }

        # Fallback 3-way split -> invoke full Arbiter synthesis (MiniMax-M3)
        rj = runner.invoke(
            task_id=task.task_id,
            role=f"judge_s5_{mode}",
            model="minimax-m3",
            prompt=build_judge_prompt(task, ra["text"], "glm-5.3-flashx", rb["text"], "mimo-v2.6-flash"),
            disable_thinking=False,
            temperature=0.0,
            max_tokens=6144,
        )
        tok += rj["total_tokens"]
        lat += rj["latency_ms"]
        return {
            "task_id": task.task_id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "is_correct": task.evaluate(rj["text"]),
            "total_tokens": tok,
            "latency_ms": lat,
            "router_tier": "tier3_quorum_arbiter",
            "shortcircuit": False,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as ex:
        s4_rows = list(ex.map(run_s4_debate, suite))
        s5_eco_rows = list(ex.map(lambda t: run_s5_router(t, "eco"), suite))
        s5_strict_rows = list(ex.map(lambda t: run_s5_router(t, "strict"), suite))

    # ------------------------------------------------------------------
    # Step 4: Assemble All 13 Live Configurations & Compute Pareto Metrics
    # ------------------------------------------------------------------
    def summarize_rows(cfg_id: str, label: str, family: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        acc = float(np.mean([r["is_correct"] for r in rows]))
        mean_tok = float(np.mean([r["total_tokens"] for r in rows]))
        p50_lat = float(np.percentile([r["latency_ms"] for r in rows], 50))
        p95_lat = float(np.percentile([r["latency_ms"] for r in rows], 95))
        by_diff = {
            d: float(np.mean([r["is_correct"] for r in rows if r["difficulty"] == d]))
            for d in ("easy", "medium", "hard")
        }
        by_dom = {
            dm: float(np.mean([r["is_correct"] for r in rows if r["domain"] == dm]))
            for dm in ("code_patch", "scientific_symbolic", "evidence_crosscheck")
        }
        out = {
            "config_id": cfg_id,
            "label": label,
            "topology_family": family,
            "accuracy": acc,
            "mean_tokens": mean_tok,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "accuracy_by_difficulty": by_diff,
            "accuracy_by_domain": by_dom,
        }
        if any("router_tier" in r for r in rows):
            tier_cnt = Counter(r["router_tier"] for r in rows)
            out["router_tier_fractions"] = {k: v / len(rows) for k, v in tier_cnt.items()}
            t3_rows = [r for r in rows if r["router_tier"] == "tier3_hetero_gate"]
            out["tier3_disagreement_shortcircuit_rate"] = (
                float(np.mean([r["shortcircuit"] for r in t3_rows])) if t3_rows else 0.0
            )
        return out

    # Build S1 & S2 row lists from cached calls
    def make_s1_rows(m_name: str, no_think: bool, max_tok: int) -> List[Dict[str, Any]]:
        rows = []
        for t in suite:
            r = get_s1(t, m_name, no_think, max_tok)
            rows.append(
                {
                    "task_id": t.task_id,
                    "domain": t.domain,
                    "difficulty": t.difficulty,
                    "is_correct": t.evaluate(r["text"]),
                    "total_tokens": r["total_tokens"],
                    "latency_ms": r["latency_ms"],
                }
            )
        return rows

    def make_sc_rows(m_name: str, k_samples: int) -> List[Dict[str, Any]]:
        rows = []
        for t in suite:
            samples = [
                runner.invoke(
                    task_id=t.task_id,
                    role="generate_sc",
                    model=m_name,
                    prompt=t.prompt,
                    disable_thinking=True,
                    temperature=0.7,
                    max_tokens=1024,
                    slot=s_idx,
                )
                for s_idx in range(1, k_samples + 1)
            ]
            canon_map: Dict[str, str] = {}
            votes: Counter[str] = Counter()
            for sp in samples:
                ck = canonicalize_live_answer(t, sp["text"])
                canon_map[ck] = sp["text"]
                votes[ck] += 1
            winner_key = votes.most_common(1)[0][0]
            winner_txt = canon_map[winner_key]
            rows.append(
                {
                    "task_id": t.task_id,
                    "domain": t.domain,
                    "difficulty": t.difficulty,
                    "is_correct": t.evaluate(winner_txt),
                    "total_tokens": sum(sp["total_tokens"] for sp in samples),
                    "latency_ms": max(sp["latency_ms"] for sp in samples),
                }
            )
        return rows

    summaries = [
        summarize_rows("S1_QwenFlash_NoThink", "SinglePass: Qwen3.8-Flash (NoThink)", "S1_SinglePass", make_s1_rows("qwen3.8-flash", True, 1024)),
        summarize_rows("S1_DeepSeekFlash_NoThink", "SinglePass: DeepSeek-Flash (NoThink)", "S1_SinglePass", make_s1_rows("deepseek-flash", True, 1024)),
        summarize_rows("S1_MiniMax_M31", "SinglePass: MiniMax-M3.1-Flash", "S1_SinglePass", make_s1_rows("minimax-m3.1-flash", False, 6144)),
        summarize_rows("S1_GLM53_FlashX", "SinglePass: GLM-5.3-FlashX", "S1_SinglePass", make_s1_rows("glm-5.3-flashx", False, 8192)),
        summarize_rows("S1_MiMo_V26", "SinglePass: MiMo-V2.6-Flash", "S1_SinglePass", make_s1_rows("mimo-v2.6-flash", False, 8192)),
        summarize_rows("S2_SC3_QwenFlash", "SC@3: Qwen3.8-Flash (NoThink)", "S2_SelfConsistency", make_sc_rows("qwen3.8-flash", 3)),
        summarize_rows("S2_SC5_DeepSeekFlash", "SC@5: DeepSeek-Flash (NoThink)", "S2_SelfConsistency", make_sc_rows("deepseek-flash", 5)),
        summarize_rows("S3_DeepSeek_WeakQwenCritic", "GenVerify: DeepSeek-Flash + Weak Qwen Critic", "S3_GenVerifyLoop", s3_ds_weak_rows),
        summarize_rows("S3_MiniMax_WeakQwenCritic", "GenVerify: MiniMax-M3.1 + Weak Qwen Critic [Harm]", "S3_GenVerifyLoop", s3_mm_weak_rows),
        summarize_rows("S3_GLM53_WeakQwenCritic", "GenVerify: GLM-5.3-FlashX + Weak Qwen Critic [Harm]", "S3_GenVerifyLoop", s3_glm_weak_rows),
        summarize_rows("S3_DeepSeek_ToolGLMVer", "GenVerify: DeepSeek-Flash + MicroCheck + GLM-5.3 Verifier", "S3_GenVerifyLoop", s3_ds_toolglm_rows),
        summarize_rows("S3_MiniMax_ToolGLMVer", "GenVerify: MiniMax-M3.1 + MicroCheck + GLM-5.3 Verifier", "S3_GenVerifyLoop", s3_mm_toolglm_rows),
        summarize_rows("S4_HeteroDebate_3Family", "DebateDAG: GLM-5.3 x MiMo-V2.6 -> MiniMax-M3 Judge", "S4_AdversarialDebateDAG", s4_rows),
        summarize_rows("S5_ParetoRouter_Eco", "ParetoRouter-Eco (OmniHarness Fast+Mid+Gate)", "S5_ParetoAdaptiveRouter", s5_eco_rows),
        summarize_rows("S5_ParetoRouter_Strict", "ParetoRouter-Strict (OmniHarness DualProbe+HeteroGate)", "S5_ParetoAdaptiveRouter", s5_strict_rows),
    ]

    # Pareto analysis
    pts = [
        ConfigPoint(
            config_id=s["config_id"],
            label=s["label"],
            topology_family=s["topology_family"],
            accuracy=s["accuracy"],
            mean_tokens=s["mean_tokens"],
            p50_latency_ms=s["p50_latency_ms"],
            p95_latency_ms=s["p95_latency_ms"],
        )
        for s in summaries
    ]
    front2d = pareto_frontier_2d(pts)
    front2d_ids = {p.config_id for p in front2d}
    _, rank_map = fast_non_dominated_sort_3d(pts)

    for s in summaries:
        s["is_pareto_2d"] = s["config_id"] in front2d_ids
        s["pareto_3d_rank"] = rank_map[s["config_id"]]

    max_tok_ref = max(p.mean_tokens for p in pts) * 1.15
    max_lat_ref = max(p.p95_latency_ms for p in pts) * 1.15
    static_pts = [p for p in pts if p.topology_family != "S5_ParetoAdaptiveRouter"]
    hv2_static = compute_hypervolume_2d(static_pts, ref_accuracy=0.35, ref_tokens=max_tok_ref)
    hv2_global = compute_hypervolume_2d(pts, ref_accuracy=0.35, ref_tokens=max_tok_ref)
    hv3_static = compute_hypervolume_3d(
        static_pts, ref_accuracy=0.35, ref_tokens=max_tok_ref, ref_latency_ms=max_lat_ref
    )
    hv3_global = compute_hypervolume_3d(
        pts, ref_accuracy=0.35, ref_tokens=max_tok_ref, ref_latency_ms=max_lat_ref
    )
    knee = find_knee_point_2d(pts)

    payload = {
        "benchmark_name": "CrossVerify-Live-45 (100% Real Multi-Family LLM API Evaluation)",
        "protocol": "Anthropic /v1/messages",
        "num_tasks": len(suite),
        "num_configurations": len(summaries),
        "total_cached_unique_api_calls": len(runner._cache),
        "elapsed_seconds": round(time.perf_counter() - t_global, 2),
        "model_families_evaluated": [
            "Qwen (qwen3.8-flash)",
            "DeepSeek (deepseek-flash)",
            "MiniMax (minimax-m3.1-flash, minimax-m3)",
            "GLM (glm-5.3-flashx)",
            "MiMo (mimo-v2.6-flash)",
        ],
        "pareto_summary": {
            "hv2d_static": hv2_static,
            "hv2d_global": hv2_global,
            "hv2d_gain_pct": (hv2_global - hv2_static) / max(hv2_static, 1e-9) * 100.0,
            "hv3d_static": hv3_static,
            "hv3d_global": hv3_global,
            "hv3d_gain_pct": (hv3_global - hv3_static) / max(hv3_static, 1e-9) * 100.0,
            "knee_point_config_id": knee.config_id,
            "pareto_2d_frontier_ids": [p.config_id for p in front2d],
        },
        "markov_live_audits": [
            s3_ds_weak_audit,
            s3_mm_weak_audit,
            s3_glm_weak_audit,
            s3_ds_toolglm_audit,
            s3_mm_toolglm_audit,
        ],
        "configurations": summaries,
    }

    (results_dir / "live_llm_benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    df_rows = []
    for s in summaries:
        df_rows.append(
            {
                "config_id": s["config_id"],
                "label": s["label"],
                "family": s["topology_family"],
                "accuracy_pct": round(s["accuracy"] * 100, 2),
                "easy_pct": round(s["accuracy_by_difficulty"]["easy"] * 100, 1),
                "medium_pct": round(s["accuracy_by_difficulty"]["medium"] * 100, 1),
                "hard_pct": round(s["accuracy_by_difficulty"]["hard"] * 100, 1),
                "mean_tokens": round(s["mean_tokens"], 1),
                "p50_latency_ms": round(s["p50_latency_ms"], 1),
                "p95_latency_ms": round(s["p95_latency_ms"], 1),
                "pareto_2d": s["is_pareto_2d"],
                "rank_3d": s["pareto_3d_rank"],
            }
        )
    df = pd.DataFrame(df_rows)
    df.to_csv(results_dir / "live_llm_comparison.csv", index=False)

    print("\n=== LIVE MULTI-FAMILY LLM BENCHMARK COMPLETE ===")
    print(df.to_string(index=False))
    print(
        f"\n2D HV Gain: +{payload['pareto_summary']['hv2d_gain_pct']:.2f}% | "
        f"3D HV Gain: +{payload['pareto_summary']['hv3d_gain_pct']:.2f}% | "
        f"Knee: {knee.config_id} | Unique API Traces Logged: {len(runner._cache)}"
    )


if __name__ == "__main__":
    main()
