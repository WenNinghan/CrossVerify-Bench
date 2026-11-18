# CrossVerify-Bench

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-tests-brightgreen.svg)](tests/test_crossverify.py)
[![Live LLM Traces](https://img.shields.io/badge/Recorded%20API%20Responses-JSONL-D97706.svg)](results/live_llm_traces.jsonl)
[![Model Families](https://img.shields.io/badge/Live%20Families-Qwen%20%C2%B7%20DeepSeek%20%C2%B7%20GLM%20%C2%B7%20MiMo%20%C2%B7%20MiniMax-0F8B6D.svg)](scripts/run_live_llm_benchmark.py)

When does a reviewer fix an answer, and when does it spoil one? In archived runs on 45 questions, the same Qwen reviewer increased DeepSeek's correct count from **22 to 37** and reduced MiniMax's from **36 to 34**. A separate evaluation on **24 new questions** used one model at two levels of reasoning effort. It recorded **one fix and no regressions** across the two runs. The repository retains the individual answers, verdicts and revisions. [Archived comparisons](#results-from-the-archived-calls) · [New question bank](data/fresh_holdout.json)

<a name="independent-24-question-replication"></a>

## Single-model review on 24 new questions

We froze 24 new questions, their executable reference answers and a scoring rule before collecting the primary responses. The new bank covers 12 problem families with two instances each, including cache eviction, constrained merging, posterior probability and bitemporal records. Each domain has eight items. Both conditions use the same questions and prompt wording. [Question bank](data/fresh_holdout.json) · [Reference calculators](scripts/build_fresh_holdout.py) · [Primary protocol](results/fresh_holdout/protocol.json)

| Generator and repair effort | Initial correct | After review | Fixed | Broken | Initially correct answers rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low | 23/24 | 24/24 | 1 | 0 | 0/23 |
| High | 24/24 | 24/24 | 0 | 0 | 0/24 |

Both conditions requested `gpt-6.2-sol` through Codex. Initial answers and reviews used fresh sessions, four questions per batch; reviewers received the proposed answers and questions, and used medium reasoning effort. Only an explicit rejection triggered a fresh repair session. Scoring used exact ordered integer arrays; model sessions received neither the answer keys nor repository files. Across 25 completed sessions, the event logs showed zero tool calls. The high-effort sensitivity condition was specified after primary collection started and before its scores were inspected. [Secondary protocol](results/fresh_holdout/higher_effort/protocol.json)

In the low-effort condition, the reviewer corrected the only error. On F04, variable-step circle elimination, the proposed array `[12,9,1]` became `[12,11,3]`. The higher-effort condition began at the ceiling. Six additional review sessions checked those 24 answers without changing any. The earlier pattern of frequent false rejection did not recur here. These adjacent pairs of items, batched calls and one run per setting provide little power to compare accuracy. The original five model families and routing policies were evaluated only on the older suite; repeating them on these new questions remains open. Recorded CLI usage includes the session's instruction prefix and is useful for accounting within this run. [Primary per-question outcomes and usage](results/fresh_holdout/paired_results.json) · [Higher-effort outcomes and usage](results/fresh_holdout/higher_effort/paired_results.json)

After collection, an independent check recomputed all 24 reference arrays directly from the question text. All matched the archived keys, including the corrected F04 array.

Replay both conditions locally using the retained responses:

```bash
python scripts/run_fresh_holdout.py
python scripts/run_fresh_holdout.py --high-effort
```

## Results from the archived calls

The current offline replay consumes **984 distinct responses** from a cache of **1,122 distinct call keys**. Policies share calls. The live sample comprises 45 tasks and one archived run per policy. The latency columns reconstruct a critical path from recorded call timings and assumed local overhead. Individual call times include provider and network variability. [Raw responses](results/live_llm_traces.jsonl) · [Replay output and task-level routing decisions](results/live_llm_benchmark.json) · [All 15 configurations with tokens and latency](results/live_llm_comparison.csv)

### What changed after review?

| Generator and reviewer | Correct before → after (of 45) | Initially correct answers rejected | Correct → wrong | Wrong → correct | Mean tokens per task |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek + Qwen | 22 → 37 | 12/22 | 1 | 16 | 2,053 |
| MiniMax + Qwen | 36 → 34 | 26/36 | 5 | 3 | 3,729 |
| GLM + Qwen | 44 → 43 | 36/44 | 1 | 0 | 6,769 |
| DeepSeek + microcheck and GLM | 22 → 42 | 0/22 | 0 | 20 | 2,000 |
| MiniMax + microcheck and GLM | 36 → 40 | 0/36 | 0 | 4 | 2,669 |

The same critic improved DeepSeek’s endpoint accuracy and reduced MiniMax’s and GLM’s on this suite. Positive and negative flips can cancel. In the MiniMax comparison, an exact two-sided paired binomial test on the eight discordant tasks gives *p* = 0.727. Zero observed corruptions in the microcheck + GLM configurations describes those runs. These configurations change both the reviewer and the guardrail, so each component’s separate contribution remains unmeasured.

Of 219 distinct verifier calls used, 203 yielded an extractable JSON object with an exact ACCEPT or REJECT verdict. Of the remaining 16, 14 were processed by the text fallback and two returned a parsed JSON object without an exact verdict. All 16 were rejected by the evaluator. The cache key omits the token cap and system prompt, and historical records omit provider stop reasons. The original token cap and truncation rate cannot be verified from the archive alone.

Two paired records illustrate a correct answer changing during repair. In `live_sci_easy_03`, GLM initially returned the fourth-order units vector `[9, 0, -1, -3]` and changed it to `[3, 0, -1, -1]` after review. In `live_evid_easy_01`, MiniMax preserved the correct power value of 152 but changed the accepted schedule `[0,1,2]` to `[2,0,1]`. Both can be found by task and configuration in the saved harm cases. [Paired records](results/live_llm_benchmark.json)

![Empirical verification audit](assets/live_llm_verification_harm_audit.png)

### How much did routing cost?

| Policy | Correct (of 45) | Mean tokens per task |
| --- | ---: | ---: |
| Single-pass GLM | 44 | 2,094 |
| Three-family debate | 43 | 4,958 |
| Router Eco | 43 | 1,560 |
| Router Strict | 45 | 2,446 |

Relative to debate, Eco used 68.5% fewer tokens at the same observed endpoint count. Strict used 50.7% fewer tokens and answered two more tasks correctly. The first consensus gate accepted 21/21 correct answers for Eco and 15/15 for Strict. These subsets overlap. A correct outcome on all 45 tasks is an observation about this suite.

The first gate uses partial invariants and agreement across models. Structured tasks compare normalized JSON. Code tasks compare identical source directly or probe behavior on eight generated inputs. The outer folds fit prompt-complexity quantiles on 36 tasks and apply them to the other nine. The comparison uses variants from the same task families in calibration and evaluation. Separate task families, repeated model calls and paired ablations would be needed to estimate transfer, sampling variability and the value of each routing mechanism.

![Live multi-family cost and accuracy](assets/live_llm_pareto_frontier.png)

---

## Cost–Error Probability Accounting & Multi-Round Stopping (`crossverify/markov_theory.py`)

### 1. Single-Round Review Accounting ($`T = 1`$)
For a single round of review and conditional repair from base accuracy $`p_0`$, let $`\alpha = \mathbb{P}(\text{reject} \mid C)`$ be the critic's Type-I false-rejection rate, $`\beta = \mathbb{P}(\text{accept} \mid I)`$ its Type-II false-acceptance rate, $`q_{I\to C}`$ the repair recovery rate on rejected errors, and $`q_{C\to I}`$ the corruption rate when a valid answer is falsely rejected. By elementary total probability:

```math
p_1 = p_0(1 - \alpha q_{C\to I}) + (1 - p_0)(1 - \beta) q_{I\to C}, \qquad \Delta p_1 = (1 - p_0)(1 - \beta) q_{I\to C} - p_0 \alpha q_{C\to I}
```

The net gain is positive exactly when $(1-p_0)(1-\beta)q_{I\to C} > p_0\alpha q_{C\to I}$. This form also covers perfect acceptance or zero corruption.

### 2. Multi-Round Stopping Rules & Expected Token Recurrence ($`T > 1`$)
When iterating beyond a single round ($`T > 1`$), state transition recurrences distinguish two stopping rules:
1. **Unconditional Re-Review (`early_stop=False`):** Re-evaluating every round with effective corruption $`u = \alpha q_{C\to I}`$ and recovery $`v = (1 - \beta) q_{I\to C}`$ converges when $`0 < u+v < 2`$ to $`p^{*} = v / (u + v)`$:

```math
p_T = p^{*} + (p_0 - p^{*})(1 - u - v)^T
```

2. **Early-Stop on Acceptance (`early_stop=True`):** Exiting immediately once the verifier accepts induces a $`2\times 2`$ open-candidate recurrence $`\mathbf{s}_t = Q^t [p_0, \; 1 - p_0]^\top`$, determining expected multi-round token expenditure:

```math
Q = \begin{pmatrix} \alpha(1 - q_{C\to I}) & (1 - \beta) q_{I\to C} \\ \alpha q_{C\to I} & (1 - \beta)(1 - q_{I\to C}) \end{pmatrix}, \qquad \mathbb{E}[\mathrm{Cost}_T^{\mathrm{early}}] = c_{\mathrm{gen}} + \sum_{t=0}^{T-1}\big(c_{\mathrm{ver}}\mathbf{1}^\top\mathbf{s}_t + c_{\mathrm{rep}}\mathbf{1}^\top Q\mathbf{s}_t\big).
```

![Verifier Sensitivity and Multi-Round Trajectories](assets/verifier_phase_diagram.png)

---

## Part II: Controlled Parametric Multi-Round Simulation (180 Tasks × 28 Configs × 100 Seeds)

To inspect multi-round $`T \in \{1, 2, 3\}`$ stopping recurrences and variation under a specified generative model beyond single-round live API budgets, `crossverify/tasks.py` and `CalibratedReplayBackend` (`crossverify/topologies.py`) evaluate a **180-task deterministic suite** (`60 Easy / 60 Medium / 60 Hard` across `code_patch`, `scientific_symbolic`, and `evidence_crosscheck`, with zero-oracle `runtime_microcheck` partial assertions such as AST parse, molar conservation $`M=0, T=-1`$, and Jacobian/dispersion sign consistency $`\det J`$, $`\max\,\mathrm{Re}(\lambda)`$) across **28 configurations × 100 Monte-Carlo seeds**:

- **Multi-Round Drift Verification:** Empirical `GenVerify@T` trajectories match the closed-form multi-round recurrence within **`0.177 percentage points`** (60,000-trial rollouts) and **`0.424 percentage points`** (180-task suite), illustrating cumulative degradation (`79.70% -> 74.84%` at $`T=1`$ -> `69.74%` at $`T=3`$) when an uncalibrated critic is iterated without early-exit guardrails.
- **Cost–Latency–Accuracy Trade-off:** Across 28 configurations, combining deterministic partial-oracle guardrails with disagreement-gated escalation expands **2D Cost–Accuracy Hypervolume by `+7.11%`** and **3D `(Accuracy, Cost, P95 Latency)` Hypervolume by `+7.22%`**, cutting token usage by **`52%–70%`** relative to static debate DAGs.

![Controlled Parametric Stress Test Pareto Frontier](assets/pareto_frontier_cost_vs_accuracy.png)

---

## Quickstart & Zero-Key Reproducibility

### Installation

```bash
pip install -e ".[dev]"
```

### 1. Replay the archived responses offline

Use the offline flag to prevent provider calls even when credentials are configured. The command recomputes the JSON and CSV outputs and preserves the trace file. Historical cache entries can include responses to earlier prompts. Only exact keys requested by the current replay contribute to its metrics.

```bash
pytest -v
python scripts/run_live_llm_benchmark.py --offline
python scripts/generate_live_figures.py
```

### 2. Run Against a Live Messages-Compatible Endpoint

```bash
export CROSSVERIFY_API_KEY="your-api-key"
export CROSSVERIFY_BASE_URL="https://api.anthropic.com"
python scripts/run_live_llm_benchmark.py
```

### 3. Run the 28-Config × 180-Task Multi-Round Simulation

```bash
python scripts/run_benchmark.py
python scripts/generate_figures.py
```

---

## References (Classical Fault-Tolerance & Testing Lineage)

1. Avizienis, A., & Chen, L. (1977). On the implementation of N-version programming for software fault-tolerance during execution. *COMPSAC*, 149–155.
2. Knight, J. C., & Leveson, N. G. (1986). An experimental evaluation of the assumption of independence in multiversion programming. *IEEE Transactions on Software Engineering*, SE-12(1), 96–109.
3. McKeeman, W. M. (1998). Differential testing for software. *Digital Technical Journal*, 10(1), 100–107.
4. Claessen, K., & Hughes, J. (2000). QuickCheck: A lightweight tool for random testing of Haskell programs. *ACM SIGPLAN Notices (ICFP)*, 35(9), 268–279.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 WenNinghan.
