# Copyright (C) 2026 花吹雪また一年P
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experimental score-only NRPA/GNRPA and stochastic rollout controls."""
from __future__ import annotations

import math
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from GakumasCLI.scenario.exam_session import SCORE_OBJECTIVE_VERSION, greedy_policy

DRINK_KEY_TURNS = not os.environ.get('GKMS_NATIVE_DRINK_KEY', '').startswith('p')
DRINK_TURNS_MAX = 15


def _turns_left(node):
    left = (int(node.config.limit_turn) + int(getattr(node.state, 'extra_turns', 0) or 0)
            - int(node.turn) + 1)
    return 0 if left < 0 else (DRINK_TURNS_MAX if left > DRINK_TURNS_MAX else left)


@dataclass
class Completion:
    session: object
    path: tuple
    trace: tuple
    objective: object = None

    @property
    def score(self):
        return int(self.session.state.score)

    @property
    def value(self):
        return self.score if self.objective is None else self.objective


@dataclass(frozen=True)
class ScorePrior:
    weights: dict
    objective_version: str = SCORE_OBJECTIVE_VERSION
    sources: tuple = ()


def save_prior(prior, path):
    document = dict(format='gkms.score_policy_prior.v1', objective_version=prior.objective_version,
                    sources=list(prior.sources), weights=[[list(k), v] for k, v in prior.weights.items()])
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')


def load_prior(path):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if (document.get('format') != 'gkms.score_policy_prior.v1'
            or document.get('objective_version') != SCORE_OBJECTIVE_VERSION):
        raise ValueError('unsupported prior format or objective_version')
    weights = {tuple(key): float(value) for key, value in document['weights']}
    if not all(math.isfinite(v) for v in weights.values()):
        raise ValueError('prior weights must be finite')
    return ScorePrior(weights, sources=tuple(document.get('sources', ())))


_FEATURE_CACHE = {}


def _card_feature(node, card):
    """A coarse description of a card: what it does, what it costs, how good it
    is.
    """
    key = (card['produce_card_id'], card.get('current_upgrade_count', 0))
    hit = _FEATURE_CACHE.get(key)
    if hit is not None:
        return hit
    resolved = card.get('resolved_card') or node.deck.refresh_card(card)
    effects = node.master.effects
    types = []
    for entry in resolved.get('playEffects') or ():
        row = effects.get(entry.get('produceExamEffectId')) if isinstance(entry, dict) else None
        kind = (row or {}).get('effectType')
        types.append(kind.get('number') if isinstance(kind, dict) else kind)
    cost = resolved.get('stamina') or 0
    category = resolved.get('category') or {}
    rarity = resolved.get('rarity') or {}
    feature = '%s/%s/%s/%s' % (
        '.'.join(str(t) for t in sorted(t for t in types if t is not None)),
        min(int(cost), 9),
        category.get('number') if isinstance(category, dict) else category,
        rarity.get('number') if isinstance(rarity, dict) else rarity)
    _FEATURE_CACHE[key] = feature
    return feature


def _fallback_keys(node, action, key, phase):
    """Coarser keys to try when the exact one was never seen in training."""
    kind = action[0]
    if kind == 'drink':
        if key[0] == 'dt':
            return ((phase, 'drink', key[2], key[3]), ('d2', key[2]), ('d1', phase))
        return (('d2', key[2]), ('d1', phase))
    if kind == 'end':
        return ()
    card = node.deck.hand[action[1]]
    return (('c2', key[1], key[2]), ('c1', key[1]),
            ('cf', _card_feature(node, card), phase), ('cf0', _card_feature(node, card)))


def _better(a, b):
    if b is None or a.value != b.value:
        return b is None or a.value > b.value
    if str(a.session.config.segment_type).startswith('Lesson'):
        sa, sb = int(a.session.state.stamina), int(b.session.state.stamina)
        if sa != sb:
            return sa > sb
    return a.path < b.path


def _choices(node, biased, prior=None, rules=None, rule_state=None):
    actions = sorted(node.legal_actions())
    if rules is not None:
        actions = rules.filter(rule_state, node, actions)
    preferred = greedy_policy(node, list(node.deck.hand)) if biased else None
    phase = min(3, (node.turn - 1) * 4 // max(1, node.config.limit_turn))
    keys, bias = [], []
    for action in actions:
        if action[0] == 'end':
            keys.append((phase, 'end'))
            bias.append(-1.0 if biased else 0.0)
        elif action[0] == 'drink':
            if DRINK_KEY_TURNS:
                keys.append(('dt', _turns_left(node), node.drinks[action[1]], len(node.drinks)))
            else:
                keys.append((phase, 'drink', node.drinks[action[1]], len(node.drinks)))
            bias.append(-1.0 if biased else 0.0)
        else:
            card = node.deck.hand[action[1]]
            keys.append((phase, card['produce_card_id'], card.get('current_upgrade_count', 0)))
            bias.append(2.0 if biased and action[1] == preferred else 0.0)
    if prior is not None:
        default = 2.0 * math.log(0.5)
        resolved = []
        for action, key, b in zip(actions, keys, bias):
            weight = prior.get(key)
            if weight is None:
                for fallback in _fallback_keys(node, action, key, phase):
                    weight = prior.get(fallback)
                    if weight is not None:
                        break
            resolved.append(b + (default if weight is None else weight))
        bias = resolved
    if rules is not None and rules.has_prefer:
        bias = [b + rules.prefer(node, a) for a, b in zip(actions, bias)]
    return actions, tuple(keys), tuple(bias)


def learn_prior(examples, *, sources=()):
    """Smoothed replay choice frequencies, from score-evaluated teachers."""
    selected, offered = {}, {}
    for root, path in examples:
        node = root.clone()
        for action in path:
            actions, keys, _ = _choices(node, False)
            chosen = actions.index(tuple(action))
            phase = min(3, (node.turn - 1) * 4 // max(1, node.config.limit_turn))
            for index, (offer, key) in enumerate(zip(actions, keys)):
                for level in (key,) + _fallback_keys(node, offer, key, phase):
                    offered[level] = offered.get(level, 0) + 1
                    if index == chosen:
                        selected[level] = selected.get(level, 0) + 1
            node.step(action)
        if not node.finished:
            raise ValueError('teacher route is incomplete')
    return ScorePrior({key: 2.0 * math.log((selected.get(key, 0) + 1) / (n + 2))
                       for key, n in offered.items()}, sources=tuple(sources))


def _probabilities(keys, bias, policy):
    logits = [policy.get(key, 0.0) + b for key, b in zip(keys, bias)]
    shift = max(logits)
    weights = [math.exp(x-shift) for x in logits]
    total = sum(weights)
    return [w/total for w in weights]


def policy_search(session, *, deadline=5, seed=182, variant='gnrpa',
                  transitions=None, levels=None, iterations=8, on_incumbent=None, prior=None, cancel=None,
                  initial_route=None, persist_policy=True, repetition_limit=0, stabilization=1, rules=None):
    """Return a replayable incumbent, never an optimality certificate.  ``levels`` None: by the transition budget,
    as the native search (scenario.search_context.auto_levels); a deadline alone keeps 2 -- this search makes two
    orders of magnitude fewer transitions per second than the native core's, so its seconds are no measure of them."""
    if levels is None:
        from GakumasCLI.scenario.search_context import auto_levels
        levels = auto_levels(transitions)
    if variant not in ('random', 'nrpa', 'gnrpa'):
        raise ValueError(variant)
    if levels < 1 or iterations < 1:
        raise ValueError('positive nesting parameters required')
    if repetition_limit < 0 or stabilization < 1:
        raise ValueError('repetition_limit must be >= 0 and stabilization >= 1')
    if deadline is None and transitions is None:
        raise ValueError('stochastic research requires a deadline or transition budget')
    if prior is not None and prior.objective_version != SCORE_OBJECTIVE_VERSION:
        raise ValueError('prior objective_version mismatch; re-evaluate teacher routes')
    bias_weights = dict(prior.weights) if prior is not None else None
    started = time.perf_counter()
    rng = random.Random(seed)
    stats = dict(expansions=0, rollouts=0, incomplete_rollouts=0, repetition_stops=0,
                 adaptation_choices=0, policy_peak_entries=0, incumbent_updates=0,
                 top_level_calls=0, top_level_adaptations=0,
                 objective_version=SCORE_OBJECTIVE_VERSION, search_seed=seed,
                 prior_sources=list(prior.sources) if prior is not None else [])
    best = None
    curve = []

    def stopped():
        return ((cancel is not None and cancel.is_set())
                or (deadline is not None and time.perf_counter()-started >= deadline)
                or (transitions is not None and stats['expansions'] >= transitions))

    def commit(result):
        nonlocal best
        if _better(result, best):
            best = result
            stats['incumbent_updates'] += 1
            curve.append(dict(score=best.score, seconds=time.perf_counter()-started,
                              transitions=stats['expansions']))
            if on_incumbent:
                on_incumbent(best.session)

    def playout(policy, greedy=False):
        node, path, trace = session.clone(), [], []
        rs = rules.start() if rules is not None else None
        while not node.finished:
            if not greedy and stopped():
                stats['incomplete_rollouts'] += 1
                return None
            if rs is not None:
                rules.sync(rs, node)
            actions, keys, bias = _choices(node, variant == 'gnrpa', bias_weights, rules, rs)
            if not actions:
                raise ValueError('unfinished state without legal actions')
            if greedy:
                index = greedy_policy(node, list(node.deck.hand))
                want = ('end',) if index is None else ('play', int(index))
                if rs is not None and want not in actions:
                    chosen = 0
                else:
                    chosen = actions.index(want)
            else:
                probabilities = _probabilities(keys, bias, policy)
                draw, total, chosen = rng.random(), 0.0, len(actions)-1
                for i, p in enumerate(probabilities):
                    total += p
                    if draw < total:
                        chosen = i
                        break
            trace.append((keys, bias, chosen))
            if rs is not None:
                turn, left = rules.taken(rs, node, actions[chosen])
                node.step(actions[chosen], observe=False)
                rules.after(rs, node, turn, left)
            else:
                node.step(actions[chosen], observe=False)
            stats['expansions'] += 1
            path.append(actions[chosen])
        stats['rollouts'] += 1
        result = Completion(node, tuple(path), tuple(trace),
                            None if rs is None else rules.objective(rs, node))
        commit(result)
        return result

    def adapt(policy, result):
        updated = dict(policy)
        for keys, bias, chosen in result.trace:
            if stopped():
                return policy
            probs = _probabilities(keys, bias, policy)
            for i, (key, p) in enumerate(zip(keys, probs)):
                updated[key] = updated.get(key, 0.0) + (int(i == chosen) - p)
                stats['adaptation_choices'] += 1
        stats['policy_peak_entries'] = max(stats['policy_peak_entries'], len(updated))
        return updated

    def nested(level, policy):
        if level == 0:
            return playout(policy)
        local_best, repeats = None, 0
        for _ in range(iterations):
            if stopped():
                break
            if level == 1 and stabilization > 1:
                result = None
                for _ in range(stabilization):
                    candidate = nested(0, policy)
                    if candidate is None:
                        break
                    if _better(candidate, result):
                        result = candidate
            else:
                result = nested(level-1, policy)
            if result is None:
                break
            if local_best is not None and result.path == local_best.path:
                repeats += 1
            else:
                repeats = 0
            if _better(result, local_best):
                local_best = result
            policy = adapt(policy, local_best)
            if repetition_limit and repeats >= repetition_limit:
                stats['repetition_stops'] += 1
                break
        return local_best

    stats['initial_incumbent'] = None
    if initial_route is not None:
        from GakumasCLI.scenario.solver import validate_initial_route
        supplied, supplied_path, meta = validate_initial_route(session, initial_route)
        stats['initial_incumbent'] = meta
        stats['expansions'] += meta['transitions']
        objective = None
        if rules is not None:
            objective = rules.evaluate(session, supplied_path)['objective']
        commit(Completion(supplied, supplied_path, (), objective))
    playout({}, greedy=True)
    top_policy = {}
    while not session.finished and not stopped():
        if variant == 'random':
            playout({})
            continue
        stats['top_level_calls'] += 1
        result = nested(levels, top_policy if persist_policy else {})
        if persist_policy and result is not None:
            target = best if best.trace else result
            top_policy = adapt(top_policy, target)
            stats['top_level_adaptations'] += 1
    if rules is not None:
        stats['objective'] = best.value
    stats.update(elapsed_seconds=time.perf_counter()-started, curve=curve,
                 search_complete=False,
                 stopped_reason='cancelled' if cancel is not None and cancel.is_set() else 'budget',
                 solution_status='best_within_budget',
                 budget=dict(deadline=deadline, transitions=transitions,
                             levels=levels, iterations=iterations, mode=variant,
                             persist_policy=bool(persist_policy),
                             repetition_limit=int(repetition_limit),
                             stabilization=int(stabilization)))
    return best.session, best.path, stats
