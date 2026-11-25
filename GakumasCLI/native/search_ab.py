# Copyright (C) 2026 matto49
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paired A/B of native search settings on the same exams, at the same budget.

    python -B GakumasCLI/gkms.py search-ab --arm L2:levels=2 --arm L3:levels=3 --seeds 50 --transitions 480000
    python -B GakumasCLI/gkms.py search-ab --arm L2:levels=2 --arm L3:levels=3 --deadline 0.5 --workers 8
    python -B GakumasCLI/gkms.py search-ab ... --loadouts EXAMS.jsonl --out GAMES.jsonl
    python -B GakumasCLI/gkms.py search-ab --summary GAMES.jsonl [--ref L2]

Every exam is built once per arm from the same loadout and exam seed (produce_value's H.I.F session builder) and searched
with the arm's settings; the score is the replayed route's.  Exams: by default produce/coverage_baselines.json (one
loadout per archetype) x --seeds exam seeds each; or --loadouts, a JSONL of {"group", "key", "seed", "loadout"}.
An arm is NAME:key=value,... with NativeSearch.policy_search keywords (levels, iterations, stabilization,
repetition_limit, persist_policy, variant, seed); the budget (--transitions or --deadline) is the same for every arm.

The summary compares every arm with the reference arm (--ref, default the first) game by game: wins / losses / ties,
the two-sided sign test p over the games that differ, and the median score ratio with a bootstrap 95% interval,
per group and over all games.  OUT keeps one line per exam, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "GakumasCLI" / "produce" / "coverage_baselines.json"
_KEYS = {"levels": int, "iterations": int, "stabilization": int, "repetition_limit": int, "seed": int,
         "persist_policy": lambda v: v.lower() in ("1", "true", "yes"), "variant": str}
_W = {}


def parse_arm(text):
    """NAME:key=value,... -> (name, keywords)."""
    name, _, spec = text.partition(":")
    if not name:
        raise ValueError(f"arm {text!r}: NAME:key=value,...")
    kw = {}
    for item in filter(None, spec.split(",")):
        key, _, value = item.partition("=")
        if key not in _KEYS:
            raise ValueError(f"arm {name}: unknown keyword {key!r} (one of {', '.join(_KEYS)})")
        kw[key] = _KEYS[key](value)
    return name, kw


def baseline_exams(n_seeds):
    """(group, key, exam seed, loadout) for every coverage baseline x n_seeds exam seeds."""
    doc = json.loads(BASELINES.read_text(encoding="utf-8"))
    out = []
    for kind, b in doc["baselines"].items():
        rng = random.Random(f"search-ab|{kind}")
        for _ in range(n_seeds):
            seed = rng.randrange(1, 2 ** 31 - 1)
            out.append((kind, f"{kind}#{seed}", seed, b["loadout"]))
    return out


def file_exams(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append((str(d.get("group", "")), str(d["key"]), int(d["seed"]), d["loadout"]))
    return out


def _init(lib):
    from GakumasCLI.produce import produce_value as PV
    PV._init(lib)
    _W.update(PV._W)


def _play(job):
    group, key, seed, loadout, arms, budget = job
    scores, used, failed = {}, {}, {}
    for name, kw in arms:
        kw = dict(kw)
        search_seed = kw.pop("seed", 182)
        s, _ = _W["build"](_W["master"], json.loads(json.dumps(loadout)), seed=seed)
        try:
            path, st = _W["ns"].policy_search(s, seed=search_seed, **budget, **kw)
            if st["status"] != 0 or not path:
                raise RuntimeError(f"status {st['status']}")
            scores[name] = int(_W["sv"].replay_session(s, path).state.score)
            used[name] = int(st.get("expansions") or 0)
        except Exception as exc:  # noqa: BLE001 -- a state the native layout does not take; reported, not scored
            failed[name] = str(exc)[:200]
    return {"group": group, "key": key, "seed": seed, "scores": scores, "transitions": used, "failed": failed}


def sign_p(wins, losses):
    """Two-sided exact sign test over the games that differ."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _median_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return statistics.median(xs), meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def summarize(rows, ref=None, out=sys.stdout):
    arms = list(dict.fromkeys(a for r in rows for a in r["scores"]))
    if not arms:
        print("no scored games", file=out)
        return
    ref = ref or arms[0]
    groups = sorted({r["group"] for r in rows})
    print(f"{len(rows)} exams, reference {ref}; ratio = arm / {ref} on the same exam", file=out)
    print(f"{'arm':<14}{'group':<22}{'n':>5}{'win':>6}{'loss':>6}{'tie':>6}{'p':>10}   median ratio (95% CI)", file=out)
    for arm in arms:
        if arm == ref:
            continue
        for group in groups + ["(all)"]:
            pairs = [(r["scores"][arm], r["scores"][ref]) for r in rows
                     if (group == "(all)" or r["group"] == group) and arm in r["scores"] and ref in r["scores"]]
            if not pairs:
                continue
            w = sum(a > b for a, b in pairs)
            lo = sum(a < b for a, b in pairs)
            med, c1, c2 = _median_ci([a / max(1, b) for a, b in pairs])
            print(f"{arm:<14}{group:<22}{len(pairs):>5}{w:>6}{lo:>6}{len(pairs) - w - lo:>6}{sign_p(w, lo):>10.3g}"
                  f"   {med:.3f} ({c1:.3f}-{c2:.3f})", file=out)
    failed = {}
    for r in rows:
        for arm in r.get("failed", {}):
            failed[arm] = failed.get(arm, 0) + 1
    if failed:
        print("not scored (search failed):", failed, file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], help="NAME:key=value,...")
    ap.add_argument("--seeds", type=int, default=50, help="exam seeds per coverage baseline")
    ap.add_argument("--loadouts", help="JSONL of {group, key, seed, loadout} instead of the coverage baselines")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--transitions", type=int)
    g.add_argument("--deadline", type=float)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--lib")
    ap.add_argument("--out")
    ap.add_argument("--summary", help="summarize an existing OUT and exit")
    ap.add_argument("--ref")
    a = ap.parse_args(argv)
    if a.summary:
        summarize([json.loads(x) for x in open(a.summary, encoding="utf-8") if x.strip()], a.ref)
        return 0
    if len(a.arm) < 2:
        ap.error("at least two --arm")
    if a.transitions is None and a.deadline is None:
        ap.error("--transitions or --deadline")
    arms = [parse_arm(x) for x in a.arm]
    budget = {"transitions": a.transitions} if a.transitions is not None else {"deadline": a.deadline}
    exams = file_exams(a.loadouts) if a.loadouts else baseline_exams(a.seeds)
    from GakumasCLI import paths
    lib = a.lib or os.environ.get("GKMSCLIGPU_LIB") or str(paths.NATIVE_LIB)
    rows, done = [], set()
    if a.out and os.path.exists(a.out):
        rows = [json.loads(x) for x in open(a.out, encoding="utf-8") if x.strip()]
        done = {r["key"] for r in rows}
    jobs = [(gr, k, s, lo, arms, budget) for gr, k, s, lo in exams if k not in done]
    print(f"{len(jobs)} exams to play ({len(done)} done), arms {[n for n, _ in arms]}, budget {budget}", flush=True)
    sink = open(a.out, "a", encoding="utf-8") if a.out else None
    try:
        with Pool(a.workers, initializer=_init, initargs=(lib,)) as pool:
            for k, r in enumerate(pool.imap_unordered(_play, jobs), 1):
                rows.append(r)
                if sink:
                    sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                    sink.flush()
                if k % 50 == 0:
                    print(f"  {k}/{len(jobs)}", flush=True)
    finally:
        if sink:
            sink.close()
    summarize(rows, a.ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
