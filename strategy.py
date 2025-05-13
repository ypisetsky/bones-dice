from abc import ABC
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import cache, lru_cache
import itertools
from statistics import mean
import random
from typing import Optional

@dataclass(frozen=True)
class Strategy:
    cutoffs: tuple[int, ...]

    def __lt__(self, other) -> bool:
        if not isinstance(other, Strategy):
            return False
        return self.cutoffs < other.cutoffs


@dataclass(frozen=True)
class Move:
    score: int
    num_dice_kept: int

NoneType = type(None)
@dataclass(frozen=True)
class Ruleset:
    large_straight: Optional[int] = 1000
    one_or_five: Optional[int] = 1
    three_of_a_kind: Optional[int] = 10
    three_pairs: Optional[int] = None

BaseRules = Ruleset()
WithThreePairs = Ruleset(three_pairs=1000)

@cache
def get_all_moves(dice: tuple[int, ...], ruleset: Ruleset, current_score=0, current_used = 0) -> set[Move]:
    counts = Counter(dice)
    moves = set()
    MULTIPLIERS = [0, 100, 20, 30, 40, 50, 60]
    if len(counts) == 6 and ruleset.large_straight:
        return set([Move(ruleset.large_straight + current_score, 6)])
    if list(counts.values()) == [2, 2, 2] and ruleset.three_pairs:
        return set([Move(ruleset.three_pairs + current_score, 6)])
    for i, c in counts.items():
        if c >= 3 and ruleset.three_of_a_kind:
            if c == 6:
                return set([Move(MULTIPLIERS[i] * 2 * ruleset.three_of_a_kind + current_score, 6)])
            new_counts = counts.copy()
            new_counts.subtract([i, i, i])
            moves.update(get_all_moves(new_counts.elements(), ruleset, current_score + MULTIPLIERS[i] * ruleset.three_of_a_kind, current_used + 3))
        if i == 1 or i == 5 and ruleset.one_or_five:
            new_counts = counts.copy()
            new_counts.subtract([i])
            moves.update(get_all_moves(new_counts.elements(), ruleset, current_score + MULTIPLIERS[i] * ruleset.one_or_five, current_used + 1))
    if current_score > 0:
        moves.add(Move(current_score, current_used))
    return moves

Situation = tuple[int, int] # score so far, # of dice locked 

def get_score(cache: dict[(Situation, bool), float], strategy: Strategy, ruleset: Ruleset, current_situation: Situation, can_stay=True) -> float:
    if (current_situation, can_stay) in cache:
        return cache[(current_situation, can_stay)]
    result = get_score_impl(cache, strategy, ruleset, current_situation, can_stay)
    cache[(current_situation, can_stay)] = result
    return result


def get_score_impl(cache: dict[(Situation, bool), float], strategy: Strategy, ruleset: Ruleset, current_situation: Situation, can_stay=True) -> float:
    score, dice_locked = current_situation
    if dice_locked == 6:
        return get_score(cache, strategy, ruleset, (score, 0))
    if score > strategy.cutoffs[dice_locked] and can_stay:
        return score
    all_scores = []
    for dice in itertools.product(range(1, 7), repeat=6 - dice_locked):
        best_score = 0
        for move in get_all_moves(dice, ruleset):
            new_situation = (score + move.score, dice_locked + move.num_dice_kept)
            best_score = max(best_score, get_score(cache, strategy, ruleset, new_situation))
        all_scores.append(best_score)
    return mean(all_scores)

def twiddle_strategy(strategy: Strategy) -> Strategy:
    return Strategy(
        tuple([max(249, t + random.choice([-50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 100])) for t in strategy.cutoffs])
    )

def callback(strat, ruleset, cache):
    return get_score(cache, strat, ruleset, (0, 0)), cache, strat

class StrategyFinder:
    caches: defaultdict[(Strategy, Ruleset), dict[(Situation, bool), float]]

    def __init__(self):
        self.caches = defaultdict(dict)

    def get_next_strategy(self, strategy: Strategy, ruleset: Ruleset, shards=16) -> Strategy:
        candidates = list(set([twiddle_strategy(strategy) for i in range(shards)] + [strategy]))
        rulesets = [ruleset] * len(candidates)
        caches = [self.caches[(candidate, ruleset)] for candidate in candidates]
        with ProcessPoolExecutor(max_workers=16) as pool:
            results = sorted(pool.map(callback, candidates, rulesets, caches))

        for _score, new_cache, candidate in results:
            self.caches[(candidate, ruleset)] = new_cache
        
        return results[-1][2]


def analyze_possibilities(num_dice: int, ruleset: Ruleset):
    all_used = 0
    bust = 0
    big_money = 0
    total = 0
    for dice in itertools.product(range(1, 7), repeat=num_dice):
        moves = get_all_moves(dice, ruleset)
        total += 1
        if len(moves) == 0:
            bust += 1
        elif any(move.num_dice_kept == num_dice for move in moves):
            all_used += 1
        elif any(move.score >= 500 for move in moves):
            big_money += 1
    print(f"Big Money: {big_money * 100 / total }%")
    print(f"All Used: {all_used * 100 / total }%")
    print(f"Bust: {bust * 100 / total}%")

def get_thresholds(strategy: Strategy, ruleset: Ruleset) -> list[int]:
    cache = {}
    fresh_score = get_score(cache, strategy, ruleset, (0, 0))
    thresholds = []
    for i in range(1, 6):
        low = 0
        high = 100
        while low < high - 1:
            med = (low + high) // 2
            curr_score = get_score(cache, strategy, ruleset, (med * 25, i), can_stay=False)
            if curr_score > fresh_score:
                high = med
            else:
                low = med
        thresholds.append(med * 25)
    return thresholds