"""Build the frozen 24-question holdout, including executable integer reference answers."""

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "fresh_holdout.json"


def reduced(value):
    return [value.numerator, value.denominator]


def build():
    tasks = []

    def add(domain, family, text, answer):
        tasks.append(
            dict(
                id=f"F{len(tasks)+1:02d}",
                domain=domain,
                family=family,
                prompt=text + " Return just the requested JSON array of integers.",
                answer=answer,
            )
        )

    add_algorithm_cases(add)
    add_quantitative_cases(add)
    add_record_cases(add)
    return tasks


def add_algorithm_cases(add):
    for cap, refs in [
        (3, [4, 2, 3, 2, 5, 4, 4, 3, 2, 6, 5, 4, 6, 3, 2, 3, 6]),
        (4, [3, 6, 2, 1, 2, 4, 6, 5, 3, 6, 1, 7, 1, 2, 4, 4, 5, 3, 1, 7, 2]),
    ]:
        cache = {}
        misses = 0
        for i, item in enumerate(refs):
            if item not in cache:
                misses += 1
                if len(cache) == cap:
                    del cache[min(cache, key=cache.get)]
                cache[item] = (0, i)
            cache[item] = (cache[item][0] + 1, i)
        add(
            "algorithm",
            "lfu",
            f"An empty capacity-{cap} cache serves pages {refs}, left to right. A miss inserts with frequency 1, a hit increments frequency, and either sets its last-use time to now. Before inserting into a full cache evict the resident with lowest frequency, breaking ties by least recent use. Reinserted pages start at 1. Give [miss count, resident IDs in ascending order ...].",
            [misses] + sorted(cache),
        )

    for n, step in [(11, 4), (13, 6)]:
        ring = list(range(1, n + 1))
        pos = 0
        gone = []
        k = step
        while len(ring) > 1:
            pos = (pos + k - 1) % len(ring)
            gone.append(ring.pop(pos))
            k = 1 + (gone[-1] % 5)
        add(
            "algorithm",
            "variable_josephus",
            f"Players 1 through {n} sit clockwise. Start at player 1, count them as 1, and remove the player counted as {step}. Starting at the removed player's clockwise successor, count that successor as 1 and remove the player at count 1+(previously removed ID modulo 5). Repeat this rule until one remains. Give [third removed ID, seventh removed ID, last remaining ID].",
            [gone[2], gone[6], ring[0]],
        )

    for weights in [[4, 7, 2, 6, 3, 5], [5, 2, 5, 3, 2, 7, 4]]:

        @lru_cache(None)
        def merge(i, j):
            if i == j:
                return 0, 1
            candidates = []
            for k in range(i, j):
                a, b = merge(i, k)
                c, d = merge(k + 1, j)
                candidates.append((a + c + sum(weights[i : j + 1]), b * d))
            best = min(cost for cost, _ in candidates)
            return best, sum(count for cost, count in candidates if cost == best)

        add(
            "algorithm",
            "alphabetic_merge",
            f"Leaves have ordered weights {weights}. Pick a full ordered binary parenthesization preserving that leaf order. Its cost sums the total descendant leaf weight at every internal node. Count different parenthesizations, not execution orders. Give [minimum cost, number of parenthesizations achieving it].",
            list(merge(0, len(weights) - 1)),
        )

    for residents, hospitals in [
        ([[1, 2, 3], [1, 3, 2], [2, 1], [3, 2, 1]], [[3, 2, 4, 1], [4, 1, 3, 2], [1, 3, 4, 2]]),
        ([[2, 1], [2, 3, 1], [3, 1, 2], [3, 2, 1]], [[2, 1, 4, 3], [4, 2, 1, 3], [1, 3, 2, 4]]),
    ]:
        pending = list(range(1, 5))
        choices = [x[:] for x in residents]
        held = {}
        while pending:
            i = pending.pop(0)
            if not choices[i - 1]:
                continue
            h = choices[i - 1].pop(0)
            old = held.get(h)
            if old is None or hospitals[h - 1].index(i) < hospitals[h - 1].index(old):
                held[h] = i
                if old is not None:
                    pending.append(old)
            else:
                pending.append(i)
        add(
            "algorithm",
            "deferred_acceptance",
            f"Four residents, IDs 1 to 4, propose to hospitals in preference order {residents}, one list per resident. Hospital preference lists for hospitals 1 through 3 are {hospitals}. Each hospital keeps its most preferred proposer so far, capacity one. Rejected residents continue down their lists. Return the final hospital for residents 1,2,3,4, using zero if unmatched.",
            [next((h for h, x in held.items() if x == i), 0) for i in range(1, 5)],
        )


def add_quantitative_cases(add):
    for vals, weights in [
        ([7, 1, 5, 4, 2, 8], [2, 1, 2, 3, 1, 2]),
        ([6, 2, 8, 1, 5, 3, 7], [1, 3, 2, 2, 1, 3, 2]),
    ]:
        blocks = []
        for v, w in zip(vals, weights):
            blocks.append((v * w, w, 1))
            while len(blocks) > 1 and Fraction(*blocks[-2][:2]) >= Fraction(*blocks[-1][:2]):
                b = blocks.pop()
                a = blocks.pop()
                blocks.append(tuple(i + j for i, j in zip(a, b)))
        total = sum(Fraction(a, b) * c for a, b, c in blocks)
        add(
            "quantitative",
            "isotonic_fit",
            f"Given response values {vals} with positive weights {weights}, fit nondecreasing values z_i by minimizing sum_i w_i(z_i-y_i)^2. Count distinct fitted levels; take the UNweighted sum of all fitted z_i. Give [number of levels, numerator of the sum, denominator] with the fraction reduced.",
            [len(blocks)] + reduced(total),
        )

    for start, top, steps in [(2, 5, 7), (3, 6, 8)]:
        dist = {start: Fraction(1)}
        for t in range(1, steps + 1):
            p = Fraction(1 if t % 2 else 2, 3)
            nxt = Counter()
            for x, w in dist.items():
                if x in (0, top):
                    nxt[x] += w
                else:
                    nxt[x + 1] += w * p
                    nxt[x - 1] += w * (1 - p)
            dist = nxt
        add(
            "quantitative",
            "alternating_walk",
            f"A particle starts at {start} on lattice sites 0..{top}. At each odd-numbered step it moves +1 with probability 1/3, and at each even step with probability 2/3; otherwise it moves -1. Endpoints are absorbing. After exactly {steps} steps, give the probability it is at {top} as [reduced numerator, denominator].",
            reduced(dist[top]),
        )

    for obs in [[1, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1]]:

        def score(states):
            num = (6, 4)[states[0]]
            for j, (z, o) in enumerate(zip(states, obs)):
                if j:
                    num *= ((8, 2), (3, 7))[states[j - 1]][z]
                num *= ((8, 2), (2, 8))[z][o]
            return num

        path = max(product(range(2), repeat=len(obs)), key=score)
        add(
            "quantitative",
            "viterbi",
            f"Two states 0 and 1 have initial probabilities (0.6,0.4), transition rows ((0.8,0.2),(0.3,0.7)), and emission rows for values (0,1) of ((0.8,0.2),(0.2,0.8)). Emit the first observation in the initial state. Observations are {obs}. Return the single maximum-joint-probability state path, earliest lexicographic path on a tie.",
            list(path),
        )

    for bag1, bag2, n in [((4, 6), (2, 5), 3), ((3, 4), (5, 6), 4)]:

        def positive(bag):
            good, bad = bag
            none = Fraction(comb(bad, n), comb(good + bad, n))
            return none * Fraction(1, 10) + (1 - none) * Fraction(4, 5)

        post = positive(bag1) / (positive(bag1) + positive(bag2))
        add(
            "quantitative",
            "pooled_bayes",
            f"Choose urn A or B by a fair coin. A contains {bag1[0]} marked and {bag1[1]} plain balls; B contains {bag2[0]} marked and {bag2[1]} plain. Draw {n} balls without replacement; one pooled detector returns positive with probability 4/5 when any draw is marked and 1/10 otherwise. Given a positive report, give P(urn A) as [reduced numerator, denominator].",
            reduced(post),
        )


def add_record_cases(add):
    for ballots in [
        [(4, [1, 2]), (4, [2, 4, 3]), (3, [3, 1]), (2, [4, 3]), (2, [4, 2]), (1, [3])],
        [(5, [1, 4]), (4, [2, 1]), (4, [3, 2]), (3, [4, 3]), (2, [4]), (1, [2, 3])],
    ]:
        live = set(range(1, 5))
        out = []
        while len(live) > 1:
            counts = Counter()
            for w, prefs in ballots:
                candidate = next((x for x in prefs if x in live), None)
                if candidate is not None:
                    counts[candidate] += w
            loser = min(live, key=lambda x: (counts[x], -x))
            out.append(loser)
            live.remove(loser)
        add(
            "records",
            "transfer_ballots",
            f"A ballot group (multiplicity, preference list) votes for its first remaining preference. Groups: {ballots}. Candidates are 1..4. Exhausted ballots contribute nothing. In each round eliminate the fewest-vote candidate, breaking equal-lowest ties by eliminating the HIGHEST ID. Continue to one survivor even if someone has a majority. Give [first eliminated, second eliminated, third eliminated, survivor].",
            out + sorted(live),
        )

    for rows, queries in [
        (
            [(1, 1, 4, 12), (2, 3, 5, 19), (3, 1, 9, 9), (4, 6, 8, 26), (5, 4, 11, 21)],
            [(2, 3), (3, 7), (2, 10), (5, 12), (7, 12)],
        ),
        (
            [
                (1, 2, 2, 14),
                (2, 4, 6, 17),
                (3, 2, 7, 11),
                (4, 5, 10, 25),
                (5, 4, 9, 16),
                (6, 7, 11, 30),
            ],
            [(1, 12), (4, 8), (4, 10), (6, 7), (7, 10)],
        ),
    ]:
        answer = []
        for valid, known in queries:
            eligible = [r for r in rows if r[1] <= valid and r[2] <= known]
            answer.append(max(eligible, key=lambda r: (r[1], r[2]))[3] if eligible else -1)
        add(
            "records",
            "bitemporal_asof",
            f"A price version is (row ID, effective date, published date, price). Versions: {rows}. A query (valid day, knowledge day) considers effective dates at most valid day and published dates at most knowledge day; among eligible rows choose greatest effective date, then latest publication. Return one price per query in order, -1 if none. Queries: {queries}.",
            answer,
        )

    for events in [
        [
            (1, "P", 5),
            (2, "R", 1),
            (2, "P", 7),
            (3, "R", 1),
            (4, "P", 9),
            (5, "R", 7),
            (7, "P", 8),
            (8, "R", 4),
        ],
        [
            (1, "R", 2),
            (2, "P", 11),
            (3, "P", 6),
            (4, "R", 2),
            (5, "R", 2),
            (1, "P", 9),
            (6, "P", 4),
            (7, "R", 3),
            (8, "R", 6),
        ],
    ]:
        seen = set()
        active = {}
        successes = 0
        for eid, kind, v in events:
            if eid in seen:
                continue
            seen.add(eid)
            if kind == "P":
                active[eid] = v
            elif v in active:
                active.pop(v)
                successes += 1
        add(
            "records",
            "idempotent_ledger",
            f"Process these events in arrival order: {events}. Each triple is (event ID, kind, value). Mark each first-seen event ID seen even if its operation fails, and ignore later duplicate IDs. P creates a payment of that amount; R reverses the still-active payment whose event ID equals value, doing nothing if unavailable. Return [total active amount, number of active payments, number of successful reversals].",
            [sum(active.values()), len(active), successes],
        )

    for table in [
        [[9, 5, 1, 7], [7, 2, 6, 1], [6, 5, 4, 3], [8, 1, 5, 4]],
        [[8, 3, 7, 6, 5], [4, 7, 2, 5, 7], [7, 8, 5, 1, 5], [6, 4, 8, 2, 7], [6, 8, 3, 5, 3]],
    ]:
        n = len(table)
        perms = list(permutations(range(n)))
        scores = [sum(table[i][j] for i, j in enumerate(p)) for p in perms]
        best = max(scores)
        winners = [p for p, v in zip(perms, scores) if v == best]
        add(
            "records",
            "record_link",
            f"Match each row to a distinct column using similarity weights {table}. Indices start at 1. Maximize total weight; if several assignments tie use the lexicographically smallest sequence of column IDs in row order. Give [maximum total, count of maximizing assignments, chosen columns in row order ...].",
            [best, len(winners)] + [j + 1 for j in winners[0]],
        )


if __name__ == "__main__":
    BANK.parent.mkdir(exist_ok=True)
    BANK.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf8")
