from strategy import get_all_moves
from strategy import Ruleset
from strategy import Move

def test_empty_ruleset():
    ruleset = Ruleset(large_straight=None, one_or_five=None, three_of_a_kind=None, three_pairs=None)

    dice_sets = [
        (1,2,3,4,5,6),
        (1,),
        (1,1,2,2,3,3),
        (1,1,2,2,2,2),
        (2,4,6,2,4,3),
        (1,1,1,1,1,1),
    ]
    for dice in dice_sets:
        assert get_all_moves(dice, ruleset) == set()

def test_triples_only():
    ruleset = Ruleset(large_straight=None, one_or_five=None, three_of_a_kind=10, three_pairs=None)

    bad_dice_sets = [
        (1,2,3,4,5,6),
        (1,),
        (1,1,2,2,3,3),
        # (1,1,2,2,2,2),
        (2,4,6,2,4,3),
        # (1,1,1,1,1,1),
    ]
    for dice in bad_dice_sets:
        assert get_all_moves(dice, ruleset) == set()

    assert Move(1000, 3) in get_all_moves((1,1,1,2), ruleset)
    assert set([Move(2000, 6)]) == get_all_moves((1,1,1,1,1,1), ruleset)
    assert set([Move(600, 6)]) == get_all_moves((3,3,3,3,3,3), ruleset)
    
    # Two different three-of-a-kinds at once
    assert Move(700, 6) in get_all_moves((3,3,3,4,4,4), ruleset)


def test_large_straight():
    ruleset = Ruleset(large_straight=1000, one_or_five=None, three_of_a_kind=None, three_pairs=None)

    bad_dice_sets = [
        # (1,2,3,4,5,6), 
        (1,),
        (1,1,2,2,3,3),
        (1,1,2,2,2,2),
        (2,4,6,2,4,3),
        (1,1,1,1,1,1),
    ]

    for dice in bad_dice_sets:
        assert get_all_moves(dice, ruleset) == set()

    assert Move(1000, 6) in get_all_moves((1,2,3,4,5,6), ruleset)
    assert Move(1000, 6) in get_all_moves((6,2,3,4,5,1), ruleset)


def test_three_pairs():
    ruleset = Ruleset(large_straight=None, one_or_five=None, three_of_a_kind=None, three_pairs=1000)

    bad_dice_sets = [
        (1,2,3,4,5,6), 
        (1,),
        # (1,1,2,2,3,3),
        (1,1,2,2,2,2),
        (2,4,6,2,4,3),
        (1,1,1,1,1,1),
    ]

    for dice in bad_dice_sets:
        assert get_all_moves(dice, ruleset) == set()

    assert Move(1000, 6) in get_all_moves((1,1,2,2,3,3), ruleset)
    assert Move(1000, 6) in get_all_moves((4,2,2,4,3,3), ruleset)


def test_ones_fives():
    ruleset = Ruleset(large_straight=None, one_or_five=1, three_of_a_kind=None, three_pairs=None)

    bad_dice_sets = [
        (2,),
        (3,4,6,),
        (2,2,2),
        (2,2,4,4,6,6),
    ]

    for dice in bad_dice_sets:
        assert get_all_moves(dice, ruleset) == set()

    assert Move(100, 2) in get_all_moves((5,5), ruleset)
    assert Move(200, 3) in get_all_moves((1,5,5,2), ruleset)
    assert Move(100, 1) in get_all_moves((1,5,5,2), ruleset)
    assert Move(100, 2) in get_all_moves((1,5,5,2), ruleset)

def test_ones_fives_triples():
    ruleset = Ruleset(one_or_five=1, three_of_a_kind=10)

    assert Move(100, 1) in get_all_moves((1,1,1), ruleset)
    assert Move(200, 2) in get_all_moves((1,1,1), ruleset)
    assert Move(300, 3) in get_all_moves((1,1,1), ruleset)
    assert Move(1000, 3) in get_all_moves((1,1,1), ruleset)

    assert Move(50, 1) in get_all_moves((5,2,2,2), ruleset)
    assert Move(200, 3) in get_all_moves((5,2,2,2), ruleset)
    assert Move(250, 4) in get_all_moves((5,2,2,2), ruleset)



if __name__ == "__main__":
    test_empty_ruleset()
    test_triples_only()
    test_large_straight()
    test_three_pairs()
    test_ones_fives()
    test_ones_fives_triples()