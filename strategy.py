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
    """
    We assume that the strategy in Bones boils down to "when do I pass?".

    Thus we represent a strategy as the maximum amount we will roll again
    for each number of locked dice (0, 1, 2, 3, 4, and 5).
    """
    cutoffs: tuple[int, ...]

    def __lt__(self, other) -> bool:
        if not isinstance(other, Strategy):
            return False
        return self.cutoffs < other.cutoffs


@dataclass(frozen=True)
class Move:
    """
    A move has some point value, and uses a certain number of dice. This is all
    we really need to care about for it.
    """
    score: int
    num_dice_kept: int

@dataclass(frozen=True)
class Ruleset:
    """
    A Ruleset represents the point values (and which scoring patterns are enabled).
    If None is set for a given scoring pattern, that scoring pattern is disabled.
    Otherwise, the value is multiplied by the inherent score of the dice involved:

    For one_or_five and three_of_a_kind:
    - a 1 has an inherent value of 100
    - Other numbers have an inherent value of 10 * the number

    For other scoring patterns: the inherent score is 1

    large_straight is rolling 1,2,3,4,5,6 on 6 dice
    one_or_five is scoring for any ones or fives
    three_of_a_kind is scoring for a three-of-a-kind
    three_pairs is scoring for three pairs (NOT a quadruple and a pair).
    """
    large_straight: Optional[int] = 1000
    one_or_five: Optional[int] = 1
    three_of_a_kind: Optional[int] = 10
    three_pairs: Optional[int] = None

BaseRules = Ruleset()
WithThreePairs = Ruleset(three_pairs=1000)

# @cache
def get_all_moves(dice: tuple[int, ...], ruleset: Ruleset, current_score=0, current_used = 0) -> set[Move]:
    """
    Finds all* of the moves that could be done with the given set of dice rolls and the given rule set.
    Can also be called recursively after some dice have already been used. If so, current_score and current_used
    should be nonzero and represent the score and number of dice already used.

    Using no dice is not valid. However, using no dice when we've already used some dice is valid.
    """
    counts = Counter(dice)
    moves = set()
    MULTIPLIERS = [0, 100, 20, 30, 40, 50, 60]

    # If all 6 numbers are different, we have a large straight. Assume that that's the best thing we could do.
    if len(counts) == 6 and ruleset.large_straight:
        return set([Move(ruleset.large_straight + current_score, 6)])
    
    # If there are three different numbers, each of which appears twice, we have three pairs. Also assume that
    # it's the best we can do.
    if list(counts.values()) == [2, 2, 2] and ruleset.three_pairs:
        return set([Move(ruleset.three_pairs + current_score, 6)])
    
    for i, c in list(counts.items()):
        if c >= 3 and ruleset.three_of_a_kind:
            if c == 6:
                # Assume we'll never roll more than 6 dice, and 6 of a kind is as good as we can do.
                return set([Move(MULTIPLIERS[i] * 2 * ruleset.three_of_a_kind + current_score, 6)])
            
            # We'll try using this three of a kind and then see if we can't form anything with the remaining dice in the roll.
            new_counts = counts.copy()
            new_counts.subtract([i, i, i])
            moves.update(get_all_moves(new_counts.elements(), ruleset, current_score + MULTIPLIERS[i] * ruleset.three_of_a_kind, current_used + 3))
        if (i == 1 or i == 5) and counts.get(i, 0) >= 1 and ruleset.one_or_five:
            # Other than 3 of a kind, ones and fives can score
            new_counts = counts.copy()
            new_counts.subtract([i])
            moves.update(get_all_moves(new_counts.elements(), ruleset, current_score + MULTIPLIERS[i] * ruleset.one_or_five, current_used + 1))
    if current_score > 0:
        # If we were looking for a result after having already used some dice, then that's a good turn.
        # However, doing nothing at all is not a valid move for a set of dice.
        moves.add(Move(current_score, current_used))
    return moves

Situation = tuple[int, int] # score so far, # of dice locked 

def get_score(cache: dict[(Situation, bool), float], strategy: Strategy, ruleset: Ruleset, current_situation: Situation, can_stay=True) -> float:
    """
    Gets the expected score following the given strategy under the given rules, starting in the
    situation described in current_situation. If can_stay is True, then it means that it's allowed
    to pass right away. Otherwise at least one roll of the dice is required.

    The strategy is used to decide whether to pass or not, depending on the situation (current score + number of dice locked).
    This function uses the average result out of all possible rolls of the dice. In order to calculate 
    """
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
    """
    Given a strategy, creates a new strategy with cutoffs that are similar but slightly different.
    """
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