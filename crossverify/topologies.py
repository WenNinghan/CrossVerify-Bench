"""Executable Multi-Agent Cross-Verification Topologies & Pareto-Adaptive Router.

Implements five orchestration families (S1-S5) supporting both:
    1. ``CalibratedReplayBackend``: Deterministic-seed calibrated multi-agent execution
       operating on real candidate patches/symbolic JSON strings, real AST/constraint
       runtime micro-checks, and realistic token/latency distributions.
    2. ``LiveOpenAIBackend``: Pluggable OpenAI-compatible HTTP chat completion client
       (compatible with OpenAI, vLLM, SGLang, Ollama, DeepSeek) for live evaluation.

Topologies
----------
- ``S1``: ``SinglePass`` (direct single-agent generation)
- ``S2``: ``SelfConsistencyVote`` (SC@K majority or confidence-weighted consensus)
- ``S3``: ``GeneratorVerifierLoop`` (GenVerify@T iterative critique & conditional repair)
- ``S4``: ``AdversarialDebateDAG`` (networkx DiGraph dual-proposer cross-exam + judge)
- ``S5``: ``ParetoAdaptiveRouter`` (OmniHarness structural + microcheck + disagreement router)
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import networkx as nx
import numpy as np

from crossverify.markov_theory import MarkovVerifierChain, VerificationParams
from crossverify.tasks import BenchmarkTask, extract_prompt_features


# ---------------------------------------------------------------------------
# Calibrated Model & Verifier Specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Capability, token expenditure, and latency profile of a generator/judge model."""

    name: str
    base_accuracy: Dict[str, float]  # keyed by "easy", "medium", "hard"
    domain_delta: Dict[str, float]   # additive domain modulation
    mean_gen_tokens: float
    mean_repair_tokens: float
    median_latency_ms: float
    latency_sigma: float
    # Distractor weights (w_d0, w_d1, w_d2): higher-capability models make fewer syntax/gross
    # errors (d0, d1) and relatively more subtle semantic errors (d2), whereas correlated
    # fast models concentrate heavily on a systematic misconception.
    distractor_probs: Tuple[float, float, float]
    # Repair transition parameters when guided by a valid critique vs false critique
    base_repair_i2c: Dict[str, float]
    base_corrupt_c2i: float

    def p0_for(self, difficulty: str, domain: str) -> float:
        raw = self.base_accuracy[difficulty] + self.domain_delta.get(domain, 0.0)
        return float(np.clip(raw, 0.02, 0.995))


@dataclass(frozen=True)
class VerifierSpec:
    """Error profile and cost of a verification agent."""

    name: str
    alpha_by_diff: Dict[str, float]  # Type-I false rejection P(reject | C)
    beta_by_diff: Dict[str, float]   # Type-II false acceptance P(accept | I) on semantic check
    use_runtime_microcheck: bool
    repair_boost: float              # Multiplicative boost to generator's q_{I->C}
    false_critique_severity: float   # Multiplier on generator's q_{C->I}
    mean_verify_tokens: float
    median_latency_ms: float
    latency_sigma: float


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "fast-8b": ModelSpec(
        name="fast-8b",
        base_accuracy={"easy": 0.885, "medium": 0.550, "hard": 0.240},
        domain_delta={"code_patch": 0.01, "scientific_symbolic": -0.01, "evidence_crosscheck": 0.00},
        mean_gen_tokens=320.0,
        mean_repair_tokens=380.0,
        median_latency_ms=235.0,
        latency_sigma=0.22,
        # Correlated misconception on d2 limits naive SC@K on Hard tasks
        distractor_probs=(0.18, 0.24, 0.58),
        base_repair_i2c={"easy": 0.62, "medium": 0.40, "hard": 0.22},
        base_corrupt_c2i=0.44,
    ),
    "mid-32b": ModelSpec(
        name="mid-32b",
        base_accuracy={"easy": 0.935, "medium": 0.735, "hard": 0.460},
        domain_delta={"code_patch": 0.015, "scientific_symbolic": 0.00, "evidence_crosscheck": -0.015},
        mean_gen_tokens=650.0,
        mean_repair_tokens=730.0,
        median_latency_ms=510.0,
        latency_sigma=0.24,
        distractor_probs=(0.14, 0.36, 0.50),
        base_repair_i2c={"easy": 0.76, "medium": 0.56, "hard": 0.38},
        base_corrupt_c2i=0.34,
    ),
    "frontier-70b": ModelSpec(
        name="frontier-70b",
        base_accuracy={"easy": 0.960, "medium": 0.830, "hard": 0.610},
        domain_delta={"code_patch": 0.015, "scientific_symbolic": 0.005, "evidence_crosscheck": -0.020},
        mean_gen_tokens=1580.0,
        mean_repair_tokens=1720.0,
        median_latency_ms=1020.0,
        latency_sigma=0.26,
        distractor_probs=(0.15, 0.40, 0.45),
        base_repair_i2c={"easy": 0.82, "medium": 0.66, "hard": 0.50},
        base_corrupt_c2i=0.30,
    ),
    "reasoning-pro": ModelSpec(
        name="reasoning-pro",
        base_accuracy={"easy": 0.975, "medium": 0.880, "hard": 0.710},
        domain_delta={"code_patch": 0.005, "scientific_symbolic": 0.020, "evidence_crosscheck": -0.025},
        mean_gen_tokens=3200.0,
        mean_repair_tokens=3000.0,
        median_latency_ms=2150.0,
        latency_sigma=0.28,
        # Different error geometry from frontier-70b (enables heterogeneous debate synergy)
        distractor_probs=(0.22, 0.43, 0.35),
        base_repair_i2c={"easy": 0.86, "medium": 0.73, "hard": 0.59},
        base_corrupt_c2i=0.24,
    ),
}


VERIFIER_REGISTRY: Dict[str, VerifierSpec] = {
    "weak-verifier": VerifierSpec(
        name="weak-verifier",
        alpha_by_diff={"easy": 0.28, "medium": 0.31, "hard": 0.35},
        beta_by_diff={"easy": 0.40, "medium": 0.45, "hard": 0.52},
        use_runtime_microcheck=False,
        repair_boost=0.76,
        false_critique_severity=1.35,
        mean_verify_tokens=540.0,
        median_latency_ms=320.0,
        latency_sigma=0.24,
    ),
    "mid-verifier": VerifierSpec(
        name="mid-verifier",
        alpha_by_diff={"easy": 0.09, "medium": 0.12, "hard": 0.16},
        beta_by_diff={"easy": 0.25, "medium": 0.31, "hard": 0.38},
        use_runtime_microcheck=False,
        repair_boost=0.95,
        false_critique_severity=1.00,
        mean_verify_tokens=840.0,
        median_latency_ms=480.0,
        latency_sigma=0.24,
    ),
    "calibrated-cross-verifier": VerifierSpec(
        name="calibrated-cross-verifier",
        alpha_by_diff={"easy": 0.030, "medium": 0.045, "hard": 0.065},
        beta_by_diff={"easy": 0.15, "medium": 0.19, "hard": 0.25},
        use_runtime_microcheck=False,
        repair_boost=1.06,
        false_critique_severity=0.75,
        mean_verify_tokens=1380.0,
        median_latency_ms=790.0,
        latency_sigma=0.25,
    ),
    "tool-augmented-verifier": VerifierSpec(
        name="tool-augmented-verifier",
        alpha_by_diff={"easy": 0.018, "medium": 0.028, "hard": 0.042},
        # Semantic beta on subtle distractors that survive runtime_microcheck; d0 & d1 are caught
        # deterministically with beta=0 by runtime_microcheck!
        beta_by_diff={"easy": 0.11, "medium": 0.15, "hard": 0.21},
        use_runtime_microcheck=True,
        repair_boost=1.14,
        false_critique_severity=0.62,
        mean_verify_tokens=1320.0,
        median_latency_ms=740.0,
        latency_sigma=0.23,
    ),
}


# ---------------------------------------------------------------------------
# Execution Trace & Backend Interface
# ---------------------------------------------------------------------------


@dataclass
class ExecutionTrace:
    """Detailed per-task multi-agent execution record."""

    task_id: str
    config_id: str
    topology_family: str
    candidate: str
    is_correct: bool
    total_tokens: float
    latency_ms: float
    llm_calls: int
    verifier_calls: int = 0
    repair_calls: int = 0
    microcheck_passed: Optional[bool] = None
    router_tier: Optional[str] = None
    dag_depth: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentBackend(Protocol):
    """Protocol for pluggable multi-agent execution backends (replay or live LLM API)."""

    def generate(
        self, task: BenchmarkTask, model_name: str, rng: np.random.Generator, variant_slot: int = 0
    ) -> Tuple[str, float, float, float]:
        """Return (candidate_str, confidence, tokens_used, latency_ms)."""
        ...

    def verify(
        self,
        task: BenchmarkTask,
        candidate: str,
        verifier_name: str,
        rng: np.random.Generator,
    ) -> Tuple[bool, str, float, float]:
        """Return (accepted, critique_msg, tokens_used, latency_ms)."""
        ...

    def repair(
        self,
        task: BenchmarkTask,
        prev_candidate: str,
        critique_msg: str,
        model_name: str,
        verifier_name: str,
        rng: np.random.Generator,
    ) -> Tuple[str, float, float, float]:
        """Return (repaired_candidate_str, confidence, tokens_used, latency_ms)."""
        ...


class CalibratedReplayBackend:
    """Calibrated trace-driven simulation backend grounded on real candidate strings & checks."""

    def __init__(
        self,
        models: Optional[Dict[str, ModelSpec]] = None,
        verifiers: Optional[Dict[str, VerifierSpec]] = None,
    ) -> None:
        self.models = models or MODEL_REGISTRY
        self.verifiers = verifiers or VERIFIER_REGISTRY

    @staticmethod
    def _sample_tokens(mean_tok: float, prompt_len: int, rng: np.random.Generator) -> float:
        prompt_overhead = 0.14 * prompt_len
        jitter = rng.normal(loc=1.0, scale=0.07)
        return float(max(64.0, (mean_tok + prompt_overhead) * np.clip(jitter, 0.78, 1.28)))

    @staticmethod
    def _sample_latency(median_ms: float, sigma: float, rng: np.random.Generator) -> float:
        return float(median_ms * np.exp(rng.normal(0.0, sigma)))

    def _pick_distractor(
        self, task: BenchmarkTask, spec: ModelSpec, rng: np.random.Generator, variant_slot: int = 0
    ) -> str:
        probs = np.array(spec.distractor_probs, dtype=np.float64)
        if variant_slot % 2 == 1:
            # Heterogeneous proposer diversity slightly shifts error mode away from d2
            probs = np.array([probs[0] + 0.05, probs[1] + 0.05, max(0.05, probs[2] - 0.10)])
        probs = probs / probs.sum()
        idx = int(rng.choice(3, p=probs))
        return task.distractors[idx]

    def generate(
        self, task: BenchmarkTask, model_name: str, rng: np.random.Generator, variant_slot: int = 0
    ) -> Tuple[str, float, float, float]:
        spec = self.models[model_name]
        p0 = spec.p0_for(task.difficulty, task.domain)
        is_c = bool(rng.random() < p0)
        if is_c:
            cand = task.ground_truth
            conf = float(np.clip(rng.normal(0.86, 0.08), 0.52, 0.99))
        else:
            cand = self._pick_distractor(task, spec, rng, variant_slot=variant_slot)
            conf = float(np.clip(rng.normal(0.64, 0.13), 0.25, 0.94))

        toks = self._sample_tokens(spec.mean_gen_tokens, len(task.prompt), rng)
        lat = self._sample_latency(spec.median_latency_ms, spec.latency_sigma, rng)
        return cand, conf, toks, lat

    def verify(
        self,
        task: BenchmarkTask,
        candidate: str,
        verifier_name: str,
        rng: np.random.Generator,
    ) -> Tuple[bool, str, float, float]:
        vspec = self.verifiers[verifier_name]
        micro_ms = 0.0
        if vspec.use_runtime_microcheck:
            ok_micro, reason = task.runtime_microcheck(candidate)
            micro_ms = 3.5
            if not ok_micro:
                # Static LLM verifier reads prompt + candidate + tool stderr and formats critique
                toks = self._sample_tokens(vspec.mean_verify_tokens, len(task.prompt), rng)
                lat = micro_ms + self._sample_latency(vspec.median_latency_ms, vspec.latency_sigma, rng)
                return False, f"tool_microcheck_failed:{reason}", toks, lat

        # Semantic LLM verification stage
        # Determine true correctness only to govern calibrated Type-I (alpha) / Type-II (beta) rates
        cand_canon = task.canonicalize(candidate)
        gt_canon = task.canonicalize(task.ground_truth)
        truly_correct = cand_canon == gt_canon

        alpha = vspec.alpha_by_diff[task.difficulty]
        beta = vspec.beta_by_diff[task.difficulty]
        toks = self._sample_tokens(vspec.mean_verify_tokens, len(task.prompt), rng)
        lat = micro_ms + self._sample_latency(vspec.median_latency_ms, vspec.latency_sigma, rng)

        if truly_correct:
            falsely_rejected = bool(rng.random() < alpha)
            if falsely_rejected:
                return False, "spurious_llm_critique:suspected_boundary_issue", toks, lat
            return True, "verified_clean", toks, lat
        else:
            falsely_accepted = bool(rng.random() < beta)
            if falsely_accepted:
                return True, "false_accept:missed_subtle_defect", toks, lat
            return False, "semantic_critique:constraint_or_invariant_mismatch", toks, lat

    def repair(
        self,
        task: BenchmarkTask,
        prev_candidate: str,
        critique_msg: str,
        model_name: str,
        verifier_name: str,
        rng: np.random.Generator,
    ) -> Tuple[str, float, float, float]:
        mspec = self.models[model_name]
        vspec = self.verifiers[verifier_name]

        was_correct = task.canonicalize(prev_candidate) == task.canonicalize(task.ground_truth)
        q_i2c, q_c2i = self.effective_repair_probs(mspec, vspec, task.difficulty, critique_msg)

        if was_correct:
            corrupted = bool(rng.random() < q_c2i)
            cand = self._pick_distractor(task, mspec, rng) if corrupted else task.ground_truth
            conf = float(np.clip(rng.normal(0.74 if corrupted else 0.88, 0.09), 0.35, 0.99))
        else:
            recovered = bool(rng.random() < q_i2c)
            cand = task.ground_truth if recovered else self._pick_distractor(task, mspec, rng)
            conf = float(np.clip(rng.normal(0.87 if recovered else 0.63, 0.10), 0.30, 0.99))

        toks = self._sample_tokens(mspec.mean_repair_tokens, len(task.prompt), rng)
        lat = self._sample_latency(mspec.median_latency_ms * 1.06, mspec.latency_sigma, rng)
        return cand, conf, toks, lat

    @staticmethod
    def effective_repair_probs(
        mspec: ModelSpec, vspec: VerifierSpec, difficulty: str, critique_msg: str = ""
    ) -> Tuple[float, float]:
        tool_bonus = 1.08 if critique_msg.startswith("tool_microcheck_failed") else 1.0
        q_i2c = float(
            np.clip(mspec.base_repair_i2c[difficulty] * vspec.repair_boost * tool_bonus, 0.05, 0.96)
        )
        q_c2i = float(
            np.clip(mspec.base_corrupt_c2i * vspec.false_critique_severity, 0.02, 0.92)
        )
        return q_i2c, q_c2i

    def exact_markov_params_for_stratum(
        self, model_name: str, verifier_name: str, difficulty: str, tasks_in_stratum: Sequence[BenchmarkTask]
    ) -> VerificationParams:
        """Derive the exact analytical Markov parameters governing a benchmark difficulty stratum."""
        mspec = self.models[model_name]
        vspec = self.verifiers[verifier_name]
        p0_vals = [mspec.p0_for(t.difficulty, t.domain) for t in tasks_in_stratum]
        p0 = float(np.mean(p0_vals))
        alpha = vspec.alpha_by_diff[difficulty]

        # In tool-augmented mode, distractors d0 & d1 fail runtime_microcheck (beta=0 on d0,d1),
        # while d2 passes runtime_microcheck and faces semantic beta_sem!
        d_probs = np.array(mspec.distractor_probs, dtype=np.float64)
        d_probs = d_probs / d_probs.sum()
        if vspec.use_runtime_microcheck:
            frac_micro_caught = float(d_probs[0] + d_probs[1])
            frac_subtle = float(d_probs[2])
            beta_eff = frac_subtle * vspec.beta_by_diff[difficulty]
            # Conditional on rejection of an incorrect answer, fraction that got tool bonus:
            p_rej_given_i = frac_micro_caught * 1.0 + frac_subtle * (1.0 - vspec.beta_by_diff[difficulty])
            w_tool = (frac_micro_caught / max(p_rej_given_i, 1e-12))
            q_tool, q_c2i = self.effective_repair_probs(
                mspec, vspec, difficulty, "tool_microcheck_failed:x"
            )
            q_sem, _ = self.effective_repair_probs(mspec, vspec, difficulty, "semantic_critique")
            q_i2c_eff = w_tool * q_tool + (1.0 - w_tool) * q_sem
        else:
            beta_eff = vspec.beta_by_diff[difficulty]
            q_i2c_eff, q_c2i = self.effective_repair_probs(mspec, vspec, difficulty, "")

        avg_prompt_len = float(np.mean([len(t.prompt) for t in tasks_in_stratum]))
        c_gen = mspec.mean_gen_tokens + 0.14 * avg_prompt_len
        c_rep = mspec.mean_repair_tokens + 0.14 * avg_prompt_len
        c_ver = vspec.mean_verify_tokens + 0.14 * avg_prompt_len
        return VerificationParams(
            p0=p0,
            alpha=alpha,
            beta=float(beta_eff),
            q_i2c=float(q_i2c_eff),
            q_c2i=float(q_c2i),
            cost_gen=c_gen,
            cost_ver=c_ver,
            cost_rep=c_rep,
        )


class LiveOpenAIBackend:
    """Pluggable OpenAI-compatible `/v1/chat/completions` client for live LLM evaluation."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: float = 60.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, float, float]:
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        lat_ms = (time.perf_counter() - t0) * 1000.0
        text = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", {})
        total_tok = float(usage.get("total_tokens", max(64, len(text) // 3)))
        return text, total_tok, lat_ms

    def generate(
        self, task: BenchmarkTask, model_name: str, rng: np.random.Generator, variant_slot: int = 0
    ) -> Tuple[str, float, float, float]:
        temp = 0.1 if variant_slot == 0 else 0.6
        txt, tok, lat = self._chat(
            model_name,
            [
                {"role": "system", "content": "Output ONLY the exact requested code or JSON."},
                {"role": "user", "content": task.prompt},
            ],
            temperature=temp,
        )
        return txt, 0.80, tok, lat

    def verify(
        self,
        task: BenchmarkTask,
        candidate: str,
        verifier_name: str,
        rng: np.random.Generator,
    ) -> Tuple[bool, str, float, float]:
        ok_m, reason = task.runtime_microcheck(candidate)
        if not ok_m:
            return False, f"tool_microcheck_failed:{reason}", 0.0, 3.0
        prompt = (
            f"Task:\n{task.prompt}\n\nCandidate Answer:\n{candidate}\n\n"
            'Reply with JSON `{"verdict": "ACCEPT" | "REJECT", "critique": "..."}`.'
        )
        txt, tok, lat = self._chat(
            verifier_name,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        accepted = '"ACCEPT"' in txt.upper() or ("ACCEPT" in txt.upper() and "REJECT" not in txt.upper())
        return accepted, txt, tok, lat

    def repair(
        self,
        task: BenchmarkTask,
        prev_candidate: str,
        critique_msg: str,
        model_name: str,
        verifier_name: str,
        rng: np.random.Generator,
    ) -> Tuple[str, float, float, float]:
        prompt = (
            f"Task:\n{task.prompt}\n\nPrevious Candidate:\n{prev_candidate}\n\n"
            f"Verifier Critique:\n{critique_msg}\n\nOutput ONLY the repaired code or JSON."
        )
        txt, tok, lat = self._chat(model_name, [{"role": "user", "content": prompt}], temperature=0.2)
        return txt, 0.80, tok, lat


class LiveAnthropicMessagesBackend:
    """Anthropic `/v1/messages` protocol client supporting multi-family gateways & thinking toggles."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 75.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.endpoint = f"{self.base_url}/messages"
        else:
            self.endpoint = f"{self.base_url}/v1/messages"
        self.timeout = timeout
        self.max_retries = max_retries

    def call_messages(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 6144,
        disable_thinking: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        if disable_thinking:
            payload["thinking"] = {"type": "disabled"}

        body_bytes = json.dumps(payload).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.endpoint,
                data=body_bytes,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "Authorization": f"Bearer {self.api_key}",
                    "anthropic-version": "2023-06-01",
                    "User-Agent": "crossverify-bench/0.3",
                },
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                lat_ms = (time.perf_counter() - t0) * 1000.0
                text = "".join(
                    blk.get("text", "") for blk in res.get("content", []) if blk.get("type") == "text"
                ).strip()
                usage = res.get("usage", {})
                in_tok = int(usage.get("input_tokens", 0))
                out_tok = int(usage.get("output_tokens", 0))
                oai_details = (
                    usage.get("billing_usage", {})
                    .get("openai_usage", {})
                    .get("completion_tokens_details")
                    or {}
                )
                reason_tok = int(oai_details.get("reasoning_tokens", 0) or 0)
                return {
                    "text": text,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "reasoning_tokens": reason_tok,
                    "total_tokens": max(1, in_tok + out_tok),
                    "latency_ms": round(lat_ms, 2),
                }
            except Exception as exc:
                last_err = exc
                time.sleep(1.2 * (attempt + 1))
        raise RuntimeError(f"Messages API failed after {self.max_retries} attempts on {model}: {last_err}")


# ---------------------------------------------------------------------------
# Five Multi-Agent Orchestration Topologies (S1 - S5)
# ---------------------------------------------------------------------------


class BaseTopology:
    """Abstract base class for multi-agent execution topologies."""

    config_id: str
    label: str
    topology_family: str

    def run_task(
        self, task: BenchmarkTask, backend: AgentBackend, rng: np.random.Generator
    ) -> ExecutionTrace:
        raise NotImplementedError


class SinglePass(BaseTopology):
    """S1: Direct single-pass generation."""

    topology_family = "S1_SinglePass"

    def __init__(self, config_id: str, label: str, model_name: str) -> None:
        self.config_id = config_id
        self.label = label
        self.model_name = model_name

    def run_task(
        self, task: BenchmarkTask, backend: AgentBackend, rng: np.random.Generator
    ) -> ExecutionTrace:
        cand, conf, tok, lat = backend.generate(task, self.model_name, rng, variant_slot=0)
        return ExecutionTrace(
            task_id=task.task_id,
            config_id=self.config_id,
            topology_family=self.topology_family,
            candidate=cand,
            is_correct=task.evaluate(cand),
            total_tokens=tok,
            latency_ms=lat,
            llm_calls=1,
            dag_depth=1,
            metadata={"confidence": conf, "model": self.model_name},
        )


class SelfConsistencyVote(BaseTopology):
    """S2: SC@K parallel candidate sampling with majority / confidence-weighted consensus."""

    topology_family = "S2_SelfConsistency"

    def __init__(
        self,
        config_id: str,
        label: str,
        model_name: str,
        k_samples: int = 5,
        weighted: bool = True,
    ) -> None:
        if k_samples < 1:
            raise ValueError("k_samples must be >= 1.")
        self.config_id = config_id
        self.label = label
        self.model_name = model_name
        self.k_samples = k_samples
        self.weighted = weighted

    def run_task(
        self, task: BenchmarkTask, backend: AgentBackend, rng: np.random.Generator
    ) -> ExecutionTrace:
        cands: List[str] = []
        canon_scores: Dict[str, float] = defaultdict(float)
        canon_counts: Counter[str] = Counter()
        canon_repr: Dict[str, str] = {}
        total_tok = 0.0
        parallel_lats: List[float] = []

        for k in range(self.k_samples):
            cand, conf, tok, lat = backend.generate(task, self.model_name, rng, variant_slot=k)
            cands.append(cand)
            total_tok += tok
            parallel_lats.append(lat)
            cn = task.canonicalize(cand)
            canon_counts[cn] += 1
            canon_scores[cn] += conf if self.weighted else 1.0
            canon_repr[cn] = cand

        best_canon = max(canon_scores.keys(), key=lambda c: (canon_scores[c], canon_counts[c]))
        chosen = canon_repr[best_canon]
        # Parallel fan-out + lightweight consensus aggregation overhead
        wall_lat = max(parallel_lats) + 12.0 + 2.5 * self.k_samples

        return ExecutionTrace(
            task_id=task.task_id,
            config_id=self.config_id,
            topology_family=self.topology_family,
            candidate=chosen,
            is_correct=task.evaluate(chosen),
            total_tokens=total_tok,
            latency_ms=wall_lat,
            llm_calls=self.k_samples,
            dag_depth=1,
            metadata={
                "k_samples": self.k_samples,
                "agreement_ratio": canon_counts[best_canon] / self.k_samples,
            },
        )


class GeneratorVerifierLoop(BaseTopology):
    """S3: Iterative Generator -> Verifier -> Conditional Repair Markov pipeline."""

    topology_family = "S3_GenVerifyLoop"

    def __init__(
        self,
        config_id: str,
        label: str,
        generator_model: str,
        verifier_name: str,
        max_rounds: int = 1,
        early_stop: bool = True,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1.")
        self.config_id = config_id
        self.label = label
        self.generator_model = generator_model
        self.verifier_name = verifier_name
        self.max_rounds = max_rounds
        self.early_stop = early_stop

    def run_task(
        self, task: BenchmarkTask, backend: AgentBackend, rng: np.random.Generator
    ) -> ExecutionTrace:
        cand, _, total_tok, total_lat = backend.generate(task, self.generator_model, rng)
        llm_calls = 1
        ver_calls = 0
        rep_calls = 0
        depth = 1

        for _ in range(self.max_rounds):
            accepted, critique, v_tok, v_lat = backend.verify(
                task, cand, self.verifier_name, rng
            )
            ver_calls += 1
            llm_calls += 1
            total_tok += v_tok
            total_lat += v_lat
            depth += 1

            if accepted:
                if self.early_stop:
                    break
                continue

            cand, _, r_tok, r_lat = backend.repair(
                task, cand, critique, self.generator_model, self.verifier_name, rng
            )
            rep_calls += 1
            llm_calls += 1
            total_tok += r_tok
            total_lat += r_lat
            depth += 1

        return ExecutionTrace(
            task_id=task.task_id,
            config_id=self.config_id,
            topology_family=self.topology_family,
            candidate=cand,
            is_correct=task.evaluate(cand),
            total_tokens=total_tok,
            latency_ms=total_lat,
            llm_calls=llm_calls,
            verifier_calls=ver_calls,
            repair_calls=rep_calls,
            dag_depth=depth,
            metadata={
                "generator": self.generator_model,
                "verifier": self.verifier_name,
                "max_rounds": self.max_rounds,
                "early_stop": self.early_stop,
            },
        )

    def analytical_predictions(
        self, tasks: Sequence[BenchmarkTask], backend: CalibratedReplayBackend
    ) -> Dict[str, Any]:
        """Compute exact analytical Markov accuracy & expected cost across (difficulty, domain) cells."""
        by_diff: Dict[str, List[BenchmarkTask]] = {"easy": [], "medium": [], "hard": []}
        for t in tasks:
            by_diff[t.difficulty].append(t)

        stratum_acc: Dict[str, float] = {}
        stratum_cost: Dict[str, float] = {}
        for diff, tlist in by_diff.items():
            if not tlist:
                continue
            by_dom: Dict[str, List[BenchmarkTask]] = defaultdict(list)
            for t in tlist:
                by_dom[t.domain].append(t)
            acc_sum = 0.0
            cost_sum = 0.0
            for _, dlist in by_dom.items():
                vparams = backend.exact_markov_params_for_stratum(
                    self.generator_model, self.verifier_name, diff, dlist
                )
                chain = MarkovVerifierChain(vparams)
                acc_sum += chain.accuracy_at_step(T=self.max_rounds, early_stop=self.early_stop) * len(dlist)
                cost_sum += chain.expected_cost(T=self.max_rounds, early_stop=self.early_stop) * len(dlist)
            stratum_acc[diff] = acc_sum / len(tlist)
            stratum_cost[diff] = cost_sum / len(tlist)

        total_n = max(len(tasks), 1)
        overall_acc = sum(stratum_acc[d] * len(by_diff[d]) for d in stratum_acc) / total_n
        overall_cost = sum(stratum_cost[d] * len(by_diff[d]) for d in stratum_cost) / total_n
        return {
            "overall_analytical_accuracy": float(overall_acc),
            "overall_analytical_cost": float(overall_cost),
            "by_difficulty_accuracy": stratum_acc,
            "by_difficulty_cost": stratum_cost,
        }


class AdversarialDebateDAG(BaseTopology):
    """S4: Explicit networkx DiGraph Dual-Proposer + Discrepancy Cross-Exam + Judge Synthesis."""

    topology_family = "S4_AdversarialDebateDAG"

    def __init__(
        self,
        config_id: str,
        label: str,
        proposer_a: str,
        proposer_b: str,
        judge_model: str,
        use_tool_in_judge: bool = False,
    ) -> None:
        self.config_id = config_id
        self.label = label
        self.proposer_a = proposer_a
        self.proposer_b = proposer_b
        self.judge_model = judge_model
        self.use_tool_in_judge = use_tool_in_judge

    @staticmethod
    def build_execution_dag(node_latencies: Dict[str, float]) -> Tuple[nx.DiGraph, float]:
        """Construct the 3-layer multi-agent execution DAG and compute critical-path latency."""
        dag = nx.DiGraph()
        for node, lat in node_latencies.items():
            dag.add_node(node, latency=lat)
        dag.add_edge("source", "proposer_A", weight=node_latencies["proposer_A"])
        dag.add_edge("source", "proposer_B", weight=node_latencies["proposer_B"])
        for p_node in ("proposer_A", "proposer_B"):
            for cx_node in ("cross_exam_A", "cross_exam_B"):
                dag.add_edge(p_node, cx_node, weight=node_latencies[cx_node])
        for cx_node in ("cross_exam_A", "cross_exam_B"):
            dag.add_edge(cx_node, "judge", weight=node_latencies["judge"])

        crit_lat = float(nx.dag_longest_path_length(dag, weight="weight"))
        return dag, crit_lat

    def run_task(
        self, task: BenchmarkTask, backend: AgentBackend, rng: np.random.Generator
    ) -> ExecutionTrace:
        # Layer 1: Parallel proposers (heterogeneous error diversity if proposer_a != proposer_b)
        hetero_slot = 1 if self.proposer_a != self.proposer_b else 0
        cand_a, conf_a, tok_a, lat_a = backend.generate(task, self.proposer_a, rng, variant_slot=0)
        cand_b, conf_b, tok_b, lat_b = backend.generate(
            task, self.proposer_b, rng, variant_slot=hetero_slot
        )

        # Layer 2: Cross-examination critiques
        ver_a = "tool-augmented-verifier" if self.use_tool_in_judge else "calibrated-cross-verifier"
        if self.proposer_a == self.proposer_b and not self.use_tool_in_judge:
            ver_a = "mid-verifier"

        acc_a, crit_a, tok_ca, lat_ca = backend.verify(task, cand_a, ver_a, rng)
        acc_b, crit_b, tok_cb, lat_cb = backend.verify(task, cand_b, ver_a, rng)

        # Layer 3: Judge synthesis & adjudication
        canon_a = task.canonicalize(cand_a)
        canon_b = task.canonicalize(cand_b)

        if canon_a == canon_b and acc_a and acc_b:
            # Proposals agree and pass cross-exam; judge confirms concisely
            _, _, j_tok_raw, j_lat_raw = backend.generate(task, self.judge_model, rng)
            j_tok = 0.55 * j_tok_raw
            j_lat = 0.55 * j_lat_raw
            final_cand = cand_a
        elif acc_a and not acc_b:
            _, _, j_tok_raw, j_lat_raw = backend.generate(task, self.judge_model, rng)
            j_tok = 0.72 * j_tok_raw
            j_lat = 0.75 * j_lat_raw
            final_cand = cand_a
        elif acc_b and not acc_a:
            _, _, j_tok_raw, j_lat_raw = backend.generate(task, self.judge_model, rng)
            j_tok = 0.72 * j_tok_raw
            j_lat = 0.75 * j_lat_raw
            final_cand = cand_b
        else:
            # Both flagged or conflicted: Judge synthesizes repaired candidate
            base_cand = cand_b if conf_b >= conf_a else cand_a
            base_crit = crit_b if conf_b >= conf_a else crit_a
            final_cand, _, j_tok, j_lat = backend.repair(
                task, base_cand, base_crit, self.judge_model, ver_a, rng
            )

        node_lats = {
            "source": 0.0,
            "proposer_A": lat_a,
            "proposer_B": lat_b,
            "cross_exam_A": lat_ca,
            "cross_exam_B": lat_cb,
            "judge": j_lat,
        }
        _, crit_path_ms = self.build_execution_dag(node_lats)
        total_tok = tok_a + tok_b + tok_ca + tok_cb + j_tok

        return ExecutionTrace(
            task_id=task.task_id,
            config_id=self.config_id,
            topology_family=self.topology_family,
            candidate=final_cand,
            is_correct=task.evaluate(final_cand),
            total_tokens=total_tok,
            latency_ms=crit_path_ms,
            llm_calls=5,
            verifier_calls=2,
            repair_calls=1,
            dag_depth=3,
            metadata={"proposers_agreed": canon_a == canon_b},
        )


class ParetoAdaptiveRouter(BaseTopology):
    """S5: OmniHarness Difficulty- & Tool-Signal-Aware Heterogeneous Routing Policy.

    Strictly zero-oracle: routes using only observable prompt structural complexity,
    runtime deterministic micro-checks, and cross-agent confidence/disagreement signals.

    Tiers:
        - ``tier1_fast``: ``fast-8b`` SinglePass + zero-token ``runtime_microcheck``;
          exits immediately if micro-check passes and confidence >= ``conf_gate``.
        - ``tier2_verified``: ``mid-32b`` or ``frontier-70b`` + ``tool-augmented-verifier``
          1-round conditional repair.
        - ``tier3_hetero_debate``: Dual heterogeneous proposers (``frontier-70b`` + partner)
          with a **Smart Disagreement Gate** — exits immediately when both proposals
          canonically match and pass ``runtime_microcheck``, invoking cross-exam + judge
          synthesis only upon discrepancy or microcheck failure.
    """

    topology_family = "S5_ParetoAdaptiveRouter"

    def __init__(
        self,
        config_id: str,
        label: str,
        tau_easy: float = 0.34,
        tau_hard: float = 0.62,
        conf_gate: float = 0.78,
        tier2_model: str = "frontier-70b",
        tier3_partner: str = "reasoning-pro",
        escalate_tier2_rejects: bool = True,
    ) -> None:
        self.config_id = config_id
        self.label = label
        self.tau_easy = tau_easy
        self.tau_hard = tau_hard
        self.conf_gate = conf_gate
        self.tier2_model = tier2_model
        self.tier3_partner = tier3_partner
        self.escalate_tier2_rejects = escalate_tier2_rejects

    def run_task(
        self, task: BenchmarkTask, backend: AgentBackend, rng: np.random.Generator
    ) -> ExecutionTrace:
        feats = extract_prompt_features(task.prompt)
        s_score = feats.complexity_score

        total_tok = 0.0
        total_lat = 2.0  # feature extraction overhead (~2 ms)
        llm_calls = 0
        ver_calls = 0
        rep_calls = 0
        depth = 0

        # --- Stage 1: Fast Probe on Low-Complexity Tasks ---
        if s_score < self.tau_easy:
            cand, conf, tok, lat = backend.generate(task, "fast-8b", rng, variant_slot=0)
            llm_calls += 1
            depth += 1
            total_tok += tok
            ok_micro, reason1 = task.runtime_microcheck(cand)
            total_lat += lat + 3.0

            if ok_micro and conf >= min(self.conf_gate, 0.74):
                return ExecutionTrace(
                    task_id=task.task_id,
                    config_id=self.config_id,
                    topology_family=self.topology_family,
                    candidate=cand,
                    is_correct=task.evaluate(cand),
                    total_tokens=total_tok,
                    latency_ms=total_lat,
                    llm_calls=llm_calls,
                    verifier_calls=0,
                    repair_calls=0,
                    microcheck_passed=True,
                    router_tier="tier1_fast",
                    dag_depth=depth,
                    metadata={"complexity_score": s_score},
                )
            if not ok_micro:
                # Zero-token microcheck caught error: perform 1 lightweight tool-guided repair!
                cand_r1, conf_r1, r_tok1, r_lat1 = backend.repair(
                    task, cand, f"tool_microcheck_failed:{reason1}", "mid-32b", "tool-augmented-verifier", rng
                )
                rep_calls += 1
                llm_calls += 1
                depth += 1
                total_tok += r_tok1
                total_lat += r_lat1
                ok_r1, _ = task.runtime_microcheck(cand_r1)
                if ok_r1 and conf_r1 >= 0.70:
                    return ExecutionTrace(
                        task_id=task.task_id,
                        config_id=self.config_id,
                        topology_family=self.topology_family,
                        candidate=cand_r1,
                        is_correct=task.evaluate(cand_r1),
                        total_tokens=total_tok,
                        latency_ms=total_lat,
                        llm_calls=llm_calls,
                        verifier_calls=0,
                        repair_calls=rep_calls,
                        microcheck_passed=True,
                        router_tier="tier1_fast",
                        dag_depth=depth,
                        metadata={"complexity_score": s_score},
                    )
            # Still uncertain -> escalate to Tier 2!

        # --- Stage 2: Calibrated Tool-Verified Generation on Medium / Escalated Tasks ---
        if s_score < self.tau_hard:
            cand2, conf2, tok2, lat2 = backend.generate(
                task, self.tier2_model, rng, variant_slot=0
            )
            llm_calls += 1
            depth += 1
            total_tok += tok2
            total_lat += lat2

            ok_m2, reason2 = task.runtime_microcheck(cand2)
            total_lat += 3.0

            if not ok_m2:
                # Zero-token runtime microcheck pre-filter caught defect directly without LLM verifier!
                critique = f"tool_microcheck_failed:{reason2}"
                accepted = False
            elif conf2 >= self.conf_gate:
                return ExecutionTrace(
                    task_id=task.task_id,
                    config_id=self.config_id,
                    topology_family=self.topology_family,
                    candidate=cand2,
                    is_correct=task.evaluate(cand2),
                    total_tokens=total_tok,
                    latency_ms=total_lat,
                    llm_calls=llm_calls,
                    verifier_calls=ver_calls,
                    repair_calls=rep_calls,
                    microcheck_passed=True,
                    router_tier="tier2_verified",
                    dag_depth=depth,
                    metadata={"complexity_score": s_score},
                )
            else:
                accepted, critique, v_tok, v_lat = backend.verify(
                    task, cand2, "tool-augmented-verifier", rng
                )
                ver_calls += 1
                llm_calls += 1
                depth += 1
                total_tok += v_tok
                total_lat += v_lat

            if accepted:
                return ExecutionTrace(
                    task_id=task.task_id,
                    config_id=self.config_id,
                    topology_family=self.topology_family,
                    candidate=cand2,
                    is_correct=task.evaluate(cand2),
                    total_tokens=total_tok,
                    latency_ms=total_lat,
                    llm_calls=llm_calls,
                    verifier_calls=ver_calls,
                    repair_calls=rep_calls,
                    microcheck_passed=True,
                    router_tier="tier2_verified",
                    dag_depth=depth,
                    metadata={"complexity_score": s_score},
                )

            # Rejected: perform 1 conditional repair, or escalate high-complexity rejects to Tier 3
            cand_rep, _, r_tok, r_lat = backend.repair(
                task, cand2, critique, self.tier2_model, "tool-augmented-verifier", rng
            )
            rep_calls += 1
            llm_calls += 1
            depth += 1
            total_tok += r_tok
            total_lat += r_lat

            ok_rep, _ = task.runtime_microcheck(cand_rep)
            if ok_rep or not self.escalate_tier2_rejects:
                return ExecutionTrace(
                    task_id=task.task_id,
                    config_id=self.config_id,
                    topology_family=self.topology_family,
                    candidate=cand_rep,
                    is_correct=task.evaluate(cand_rep),
                    total_tokens=total_tok,
                    latency_ms=total_lat,
                    llm_calls=llm_calls,
                    verifier_calls=ver_calls,
                    repair_calls=rep_calls,
                    microcheck_passed=ok_rep,
                    router_tier="tier2_verified",
                    dag_depth=depth,
                    metadata={"complexity_score": s_score},
                )

        # --- Stage 3: Heterogeneous Dual-Proposer + Smart Disagreement Gate ---
        cand_a, conf_a, tok_a, lat_a = backend.generate(task, "frontier-70b", rng, variant_slot=0)
        cand_b, conf_b, tok_b, lat_b = backend.generate(
            task, self.tier3_partner, rng, variant_slot=1
        )
        llm_calls += 2
        depth += 1
        total_tok += tok_a + tok_b
        total_lat += max(lat_a, lat_b) + 4.0

        ok_a, _ = task.runtime_microcheck(cand_a)
        ok_b, _ = task.runtime_microcheck(cand_b)

        # Smart Disagreement Gate: skip cross-exam & judge chatter if heterogeneous models agree!
        if ok_a and ok_b and task.canonicalize(cand_a) == task.canonicalize(cand_b):
            return ExecutionTrace(
                task_id=task.task_id,
                config_id=self.config_id,
                topology_family=self.topology_family,
                candidate=cand_a,
                is_correct=task.evaluate(cand_a),
                total_tokens=total_tok,
                latency_ms=total_lat,
                llm_calls=llm_calls,
                verifier_calls=ver_calls,
                repair_calls=rep_calls,
                microcheck_passed=True,
                router_tier="tier3_hetero_debate",
                dag_depth=depth,
                metadata={"complexity_score": s_score, "disagreement_gate_shortcircuit": True},
            )

        # Disagreement or microcheck failure: asymmetric tool filter + single joint adjudication
        if ok_a and not ok_b:
            if conf_a >= 0.68:
                final_c = cand_a
            else:
                acc_a, crit_a, v_tok, v_lat = backend.verify(
                    task, cand_a, "tool-augmented-verifier", rng
                )
                ver_calls += 1
                llm_calls += 1
                depth += 1
                total_tok += v_tok
                total_lat += v_lat
                if acc_a:
                    final_c = cand_a
                else:
                    final_c, _, r_tok, r_lat = backend.repair(
                        task, cand_a, crit_a, "frontier-70b", "tool-augmented-verifier", rng
                    )
                    rep_calls += 1
                    llm_calls += 1
                    depth += 1
                    total_tok += r_tok
                    total_lat += r_lat
        elif ok_b and not ok_a:
            if conf_b >= 0.68:
                final_c = cand_b
            else:
                acc_b, crit_b, v_tok, v_lat = backend.verify(
                    task, cand_b, "tool-augmented-verifier", rng
                )
                ver_calls += 1
                llm_calls += 1
                depth += 1
                total_tok += v_tok
                total_lat += v_lat
                if acc_b:
                    final_c = cand_b
                else:
                    final_c, _, r_tok, r_lat = backend.repair(
                        task, cand_b, crit_b, "frontier-70b", "tool-augmented-verifier", rng
                    )
                    rep_calls += 1
                    llm_calls += 1
                    depth += 1
                    total_tok += r_tok
                    total_lat += r_lat
        else:
            # Single joint discrepancy verification on higher-confidence proposal, fallback to partner/repair
            primary, backup = (cand_a, cand_b) if conf_a >= conf_b else (cand_b, cand_a)
            acc_p, crit_p, v_tok, v_lat = backend.verify(
                task, primary, "tool-augmented-verifier", rng
            )
            ver_calls += 1
            llm_calls += 1
            depth += 1
            total_tok += v_tok
            total_lat += v_lat

            if acc_p:
                final_c = primary
            elif ok_a and ok_b:
                # Primary rejected on subtle semantic discrepancy while backup passed microcheck:
                # repair conditioned on discrepancy critique!
                final_c, _, r_tok, r_lat = backend.repair(
                    task, backup, crit_p, "frontier-70b", "tool-augmented-verifier", rng
                )
                rep_calls += 1
                llm_calls += 1
                depth += 1
                total_tok += r_tok
                total_lat += r_lat
            else:
                final_c, _, r_tok, r_lat = backend.repair(
                    task, primary, crit_p, "frontier-70b", "tool-augmented-verifier", rng
                )
                rep_calls += 1
                llm_calls += 1
                depth += 1
                total_tok += r_tok
                total_lat += r_lat

        ok_final, _ = task.runtime_microcheck(final_c)
        return ExecutionTrace(
            task_id=task.task_id,
            config_id=self.config_id,
            topology_family=self.topology_family,
            candidate=final_c,
            is_correct=task.evaluate(final_c),
            total_tokens=total_tok,
            latency_ms=total_lat,
            llm_calls=llm_calls,
            verifier_calls=ver_calls,
            repair_calls=rep_calls,
            microcheck_passed=ok_final,
            router_tier="tier3_hetero_debate",
            dag_depth=depth,
            metadata={"complexity_score": s_score, "disagreement_gate_shortcircuit": False},
        )


def build_benchmark_configurations() -> List[BaseTopology]:
    """Build all 28 multi-agent configurations spanning topologies S1 through S5."""
    configs: List[BaseTopology] = [
        # --- S1: SinglePass (4 configs) ---
        SinglePass("S1_Fast", "SinglePass (Fast-8B)", "fast-8b"),
        SinglePass("S1_Mid", "SinglePass (Mid-32B)", "mid-32b"),
        SinglePass("S1_Frontier", "SinglePass (Frontier-70B)", "frontier-70b"),
        SinglePass("S1_Reasoning", "SinglePass (Reasoning-Pro)", "reasoning-pro"),
        # --- S2: Self-Consistency Voting SC@K (8 configs) ---
        SelfConsistencyVote("S2_SC3_Fast", "SC@3 (Fast-8B)", "fast-8b", k_samples=3),
        SelfConsistencyVote("S2_SC5_Fast", "SC@5 (Fast-8B)", "fast-8b", k_samples=5),
        SelfConsistencyVote("S2_SC7_Fast", "SC@7 (Fast-8B)", "fast-8b", k_samples=7),
        SelfConsistencyVote("S2_SC9_Fast", "SC@9 (Fast-8B)", "fast-8b", k_samples=9),
        SelfConsistencyVote("S2_SC3_Mid", "SC@3 (Mid-32B)", "mid-32b", k_samples=3),
        SelfConsistencyVote("S2_SC5_Mid", "SC@5 (Mid-32B)", "mid-32b", k_samples=5),
        SelfConsistencyVote("S2_SC7_Mid", "SC@7 (Mid-32B)", "mid-32b", k_samples=7),
        SelfConsistencyVote("S2_SC5_Frontier", "SC@5 (Frontier-70B)", "frontier-70b", k_samples=5),
        # --- S3: Generator-Verifier Iterative Loops (9 configs) ---
        # Negative result / Markov Harm configs:
        GeneratorVerifierLoop(
            "S3_GV1_Frontier_WeakVer",
            "GenVerify@1 (Frontier + WeakVer)",
            "frontier-70b",
            "weak-verifier",
            max_rounds=1,
            early_stop=True,
        ),
        GeneratorVerifierLoop(
            "S3_GV3_Frontier_WeakVer_Homog",
            "GenVerify@3-Homog (Frontier + WeakVer)",
            "frontier-70b",
            "weak-verifier",
            max_rounds=3,
            early_stop=False,
        ),
        # Same-family / Mid verification:
        GeneratorVerifierLoop(
            "S3_GV1_Fast_MidVer",
            "GenVerify@1 (Fast + MidVer)",
            "fast-8b",
            "mid-verifier",
            max_rounds=1,
        ),
        GeneratorVerifierLoop(
            "S3_GV2_Mid_MidVer",
            "GenVerify@2 (Mid + MidVer)",
            "mid-32b",
            "mid-verifier",
            max_rounds=2,
        ),
        # Calibrated Cross-Model Verification:
        GeneratorVerifierLoop(
            "S3_GV1_Mid_CalibVer",
            "GenVerify@1 (Mid + CalibVer)",
            "mid-32b",
            "calibrated-cross-verifier",
            max_rounds=1,
        ),
        GeneratorVerifierLoop(
            "S3_GV2_Mid_CalibVer",
            "GenVerify@2 (Mid + CalibVer)",
            "mid-32b",
            "calibrated-cross-verifier",
            max_rounds=2,
        ),
        GeneratorVerifierLoop(
            "S3_GV2_Frontier_CalibVer",
            "GenVerify@2 (Frontier + CalibVer)",
            "frontier-70b",
            "calibrated-cross-verifier",
            max_rounds=2,
        ),
        # Tool-Augmented Cross-Verification:
        GeneratorVerifierLoop(
            "S3_GV1_Frontier_ToolVer",
            "GenVerify@1 (Frontier + ToolVer)",
            "frontier-70b",
            "tool-augmented-verifier",
            max_rounds=1,
        ),
        GeneratorVerifierLoop(
            "S3_GV3_Frontier_ToolVer",
            "GenVerify@3 (Frontier + ToolVer)",
            "frontier-70b",
            "tool-augmented-verifier",
            max_rounds=3,
        ),
        # --- S4: Adversarial Debate DAGs (3 configs) ---
        AdversarialDebateDAG(
            "S4_Debate_Homog_Mid",
            "DebateDAG (Homog 2xMid)",
            "mid-32b",
            "mid-32b",
            "mid-32b",
            use_tool_in_judge=False,
        ),
        AdversarialDebateDAG(
            "S4_Debate_Hetero_MidFrontier",
            "DebateDAG (Hetero Mid+Frontier)",
            "mid-32b",
            "frontier-70b",
            "frontier-70b",
            use_tool_in_judge=False,
        ),
        AdversarialDebateDAG(
            "S4_Debate_Hetero_ToolJudge",
            "DebateDAG (Frontier+Reasoning + ToolJudge)",
            "frontier-70b",
            "reasoning-pro",
            "frontier-70b",
            use_tool_in_judge=True,
        ),
        # --- S5: Pareto-Adaptive Heterogeneous Routers (4 operating points) ---
        ParetoAdaptiveRouter(
            "S5_Router_Lite",
            "ParetoRouter-Lite",
            tau_easy=0.52,
            tau_hard=0.84,
            conf_gate=0.68,
            tier2_model="mid-32b",
            tier3_partner="mid-32b",
            escalate_tier2_rejects=False,
        ),
        ParetoAdaptiveRouter(
            "S5_Router_Eco",
            "ParetoRouter-Eco",
            tau_easy=0.40,
            tau_hard=0.68,
            conf_gate=0.76,
            tier2_model="mid-32b",
            tier3_partner="mid-32b",
            escalate_tier2_rejects=True,
        ),
        ParetoAdaptiveRouter(
            "S5_Router_Balanced",
            "ParetoRouter-Balanced",
            tau_easy=0.35,
            tau_hard=0.58,
            conf_gate=0.80,
            tier2_model="mid-32b",
            tier3_partner="frontier-70b",
            escalate_tier2_rejects=True,
        ),
        ParetoAdaptiveRouter(
            "S5_Router_Strict",
            "ParetoRouter-Strict",
            tau_easy=0.30,
            tau_hard=0.48,
            conf_gate=0.85,
            tier2_model="frontier-70b",
            tier3_partner="reasoning-pro",
            escalate_tier2_rejects=True,
        ),
    ]
    return configs
