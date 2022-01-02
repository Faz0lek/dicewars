from dicewars.client.game.board import Board
from typing import List, Tuple

def sort_by_first_and_get_second(dictionary: dict) -> list:
    return [pair[1] for pair in sorted(dictionary.items(), key=lambda pair: pair[0])]


def serialize_neighbourhoods(board: Board) -> List[int]:
    areas_n = len(board.areas)
    neighbourhood_dict = {(x + 1, y + 1): 0 for x in range(areas_n) for y in range(x + 1, areas_n)}
    for area in board.areas.values():
        for neighbour_name in area.get_adjacent_areas_names():
            index = (area.name, neighbour_name)
            if index in neighbourhood_dict:
                neighbourhood_dict[index] = 1
    return sort_by_first_and_get_second(neighbourhood_dict)


def serialize_board_without_neighbours(board: Board, current_player_name: int, number_of_players: int = 4) -> List[int]:
    owner_dict = {}
    dice_dict = {}

    for area in board.areas.values():
        owner_dict[area.name] = area.owner_name
        dice_dict[area.name] = area.dice

    flat_owners = sort_by_first_and_get_second(owner_dict)
    flat_dice = sort_by_first_and_get_second(dice_dict)

    largest_regions = [max([len(reg) for reg in board.get_players_regions(player)], default=0)
                       for player in range(1, number_of_players + 1)]

    current_player_one_hot = [int(player == current_player_name)
                              for player in range(1, number_of_players + 1)]

    return current_player_one_hot + flat_owners + flat_dice + largest_regions


def serialize_board_full(board: Board, current_player_name: int, number_of_players: int = 4) -> List[int]:
    return serialize_board_without_neighbours(board, current_player_name, number_of_players) + serialize_neighbourhoods(board)
