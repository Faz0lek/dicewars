from dicewars.client.game.board import Board
from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 637
OUTPUT_SIZE = 4

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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(32, OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
