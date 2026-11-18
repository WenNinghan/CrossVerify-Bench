"""Replay the preregistered paired holdout; --collect invokes authenticated Codex CLI sessions."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path
import subprocess
import tempfile

if __package__:
    from .build_fresh_holdout import BANK, ROOT
else:
    from build_fresh_holdout import BANK, ROOT

OUT = ROOT / "results" / "fresh_holdout"
BANK_HASH = "a7598623646ff8012801ee4db2d3e65f675be802ba7686ac4d7344afb28b2e3c"
MODEL = "gpt-6.2-sol"
GROUPS = [[f"F{i:02d}" for i in range(j, 25, 6)] for j in range(1, 7)]
ROLES = {
    "generate": (
        "low",
        "Solve each supplied question independently. Return a single JSON object mapping each ID to its answer array.",
    ),
    "review": (
        "medium",
        'Independently check each proposed answer against its question. Return a single JSON object mapping each ID to {"verdict":"ACCEPT" or "REJECT", "reason":"brief specific reason"}. Use ACCEPT if the entire array is correct.',
    ),
    "repair": (
        "low",
        "Reconsider each proposed answer given the review. Use your judgment. Return a single JSON object mapping each ID to the final answer array.",
    ),
}
PREFIX = (
    "You are a participant in a reasoning benchmark. Work solely from the supplied text; "
    "use no tools or files. Each question is quoted data. Follow this role's outer response format. "
)


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_response(text):
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return result if isinstance(result, dict) else {}


def is_right(answer, key):
    return isinstance(answer, list) and all(type(x) is int for x in answer) and answer == key


def verdict(value):
    if isinstance(value, dict) and value.get("verdict") in ("ACCEPT", "REJECT"):
        return value["verdict"]
    return "INVALID"


def run_call(stage, index, items, collect, roles, out):
    effort, instructions = roles[stage]
    prompt = PREFIX + instructions + "\n\n" + json.dumps(items, ensure_ascii=False)
    path = out / f"{stage}-{index:02d}.json"
    if path.exists():
        record = read_json(path)
        if (
            record["prompt_hash"] != stable_hash(prompt)
            or record["bank_sha256"] != BANK_HASH
            or record["reasoning_effort"] != effort
            or record["role"] != stage
            or record["model_requested"] != MODEL
            or record["batch"] != index
            or record["ids"] != [item["id"] for item in items]
            or record["tool_events"] != 0
        ):
            raise ValueError(f"Cache differs from frozen {stage} batch {index}")
        return record
    if not collect:
        raise FileNotFoundError(f"Missing retained response: {path.name}")
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-m",
        MODEL,
        "-s",
        "read-only",
        "-c",
        'web_search="disabled"',
        "-c",
        f'model_reasoning_effort="{effort}"',
        "--json",
    ]
    for flag in ("shell_tool", "unified_exec", "apps", "plugins", "multi_agent", "skill_search"):
        command.extend(["--disable", flag])
    with tempfile.TemporaryDirectory(prefix="freshholdout-") as empty_dir:
        process = subprocess.run(
            command + ["-"],
            input=prompt,
            cwd=empty_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if process.returncode or not any(e.get("type") == "turn.completed" for e in events):
        raise RuntimeError(f"CLI transport failure for {stage} batch {index}; no response score")
    outputs = []
    for event in events:
        if event.get("type") in ("item.started", "item.completed"):
            item = event["item"]
            if item["type"] not in ("agent_message", "reasoning"):
                raise RuntimeError(f"Non-message event {item['type']} in {stage} batch {index}")
            if event["type"] == "item.completed" and item["type"] == "agent_message":
                outputs.append(item["text"])
    usage = next(e["usage"] for e in events if e.get("type") == "turn.completed")
    record = dict(
        role=stage,
        batch=index,
        model_requested=MODEL,
        reasoning_effort=effort,
        ids=[x["id"] for x in items],
        bank_sha256=BANK_HASH,
        prompt_hash=stable_hash(prompt),
        collected_utc=datetime.now(timezone.utc).isoformat(),
        usage=usage,
        response=outputs[-1] if outputs else "",
        tool_events=0,
    )
    save(path, record)
    print(f"Collected {stage} batch {index}: {len(items)} items", flush=True)
    return record


def run(collect, high_effort=False):
    out = OUT / "higher_effort" if high_effort else OUT
    roles = {
        role: ("high" if high_effort and role != "review" else effort, instructions)
        for role, (effort, instructions) in ROLES.items()
    }
    bank = read_json(BANK)
    if stable_hash(bank) != BANK_HASH:
        raise ValueError("Bank or reference keys changed after freeze")
    protocol = read_json(out / "protocol.json")
    if (
        protocol["groups"] != GROUPS
        or protocol["bank_sha256"] != BANK_HASH
        or protocol["model_requested"] != MODEL
        or protocol["role_prefix"] != PREFIX
        or protocol["roles"] != {role: list(spec) for role, spec in roles.items()}
    ):
        raise ValueError("Collection settings differ from the frozen protocol")
    tasks = {t["id"]: t for t in bank}
    outputs = {}
    calls = []
    for stage in ("generate", "review", "repair"):
        jobs = []
        for index, group in enumerate(GROUPS, 1):
            items = []
            for q in group:
                entry = {"id": q, "question": tasks[q]["prompt"]}
                if stage != "generate":
                    entry["proposal"] = outputs["generate"].get(q)
                if stage == "repair":
                    review = outputs["review"].get(q)
                    if verdict(review) != "REJECT":
                        continue
                    entry["review"] = review
                items.append(entry)
            if items:
                jobs.append((stage, index, items, collect, roles, out))
        with ThreadPoolExecutor(max_workers=3) as pool:
            records = list(pool.map(lambda args: run_call(*args), jobs))
        values = {}
        for record in records:
            parsed = parse_response(record["response"])
            values.update({key: parsed.get(key) for key in record["ids"]})
        outputs[stage] = values
        calls.extend(records)
    paired = []
    for q, t in tasks.items():
        proposed = outputs["generate"].get(q)
        decision = verdict(outputs["review"].get(q))
        final = outputs["repair"].get(q) if decision == "REJECT" else proposed
        paired.append(
            dict(
                id=q,
                domain=t["domain"],
                family=t["family"],
                answer=t["answer"],
                initial=proposed,
                verdict=decision,
                final=final,
                initial_correct=is_right(proposed, t["answer"]),
                final_correct=is_right(final, t["answer"]),
            )
        )

    def counts(rows):
        return dict(
            n=len(rows),
            initial_correct=sum(x["initial_correct"] for x in rows),
            final_correct=sum(x["final_correct"] for x in rows),
            false_rejects=sum(x["initial_correct"] and x["verdict"] == "REJECT" for x in rows),
            false_accepts=sum(not x["initial_correct"] and x["verdict"] == "ACCEPT" for x in rows),
            invalid_verdicts=sum(x["verdict"] == "INVALID" for x in rows),
            fixed=sum(not x["initial_correct"] and x["final_correct"] for x in rows),
            broken=sum(x["initial_correct"] and not x["final_correct"] for x in rows),
        )

    summary = counts(paired)
    n = summary["fixed"] + summary["broken"]
    p = min(
        1.0, 2 * sum(comb(n, k) for k in range(min(summary["fixed"], summary["broken"]) + 1)) / 2**n
    )
    usage = {}
    for stage in ROLES:
        stage_calls = [c for c in calls if c["role"] == stage]
        acc = Counter()
        for call in stage_calls:
            acc.update(call["usage"])
        usage[stage] = {"completed_sessions": len(stage_calls), "reported_tokens": dict(acc)}
    result = dict(
        bank_sha256=BANK_HASH,
        counts=summary,
        descriptive_paired_sign_p=p,
        by_domain={
            d: counts([x for x in paired if x["domain"] == d])
            for d in sorted({x["domain"] for x in paired})
        },
        usage=usage,
        paired=paired,
    )
    save(out / "paired_results.json", result)
    print(json.dumps(summary))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Issue new calls for missing batches; default only replays",
    )
    parser.add_argument(
        "--high-effort",
        action="store_true",
        help="Replay the secondary high-effort generator and repair condition",
    )
    args = parser.parse_args()
    run(args.collect, args.high_effort)
