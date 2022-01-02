import logging
import random
from copy import deepcopy
from typing import List, Tuple

import numpy as np
from dicewars.ai.utils import (possible_attacks,
                               probability_of_successful_attack)
from dicewars.client.ai_driver import (BattleCommand, EndTurnCommand,
                                       TransferCommand)
from dicewars.client.game.area import Area
from dicewars.client.game.board import Board
from dicewars.supp_xkoste12.model import NeuralNetwork
from dicewars.ai.xkoste12.utils import serialize_board_full

import torch


MODEL_PATH = r"./dicewars/ai/xkoste12/model.pth"

class AI:
    """Naive player agent

    This agent performs all possible moves in random order
    """

    def __init__(self, player_name, board, players_order, max_transfers):
        """
        Keyword arguments:
        player_name -- Player id (int)
        board -- Board, instance of dicewars.client.game.Board
        players_order -- Order of players (list of ints)
        max_transfers -- Max transfers per turn (int)
        """

        self.player_name = player_name
        self.logger = logging.getLogger('AI xkoste12')
        self.max_transfers = max_transfers
        self.players_order = players_order
        self.board = board
        
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()

        self.MAXN_MAX_DEPTH = 3

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn (a single action)

        Keyword arguments:
        board -- Board, instance of dicewars.client.game.Board
        nb_moves_this_turn -- number of attacks made in this turn (int)
        nb_transfers_this_turn -- number of transfers made in this turn (int)
        nb_turns_this_game -- number of turns ended so far (int)
        time_left -- Time left in seconds (float)

        Returns:
        BattleCommand() || EndTurnCommand() || TransferCommand()
        """

        if time_left > 10.0:
            self.MAXN_MAX_DEPTH = 3
        elif time_left >= 2.0:
            self.MAXN_MAX_DEPTH = 2
        else:
            self.MAXN_MAX_DEPTH = 1

        self.logger.debug(f"It's my turn now. On turn: {nb_turns_this_game}.")

        self.board = board
        move = self.get_best_move(board, nb_transfers_this_turn)

        if move:
            src, tgt = move

            if board.get_area(tgt).get_owner_name() == self.player_name:
                self.logger.debug(f"Best move in this position is a transfer from {src} to {tgt}.")
                return TransferCommand(src, tgt)
            else:
                self.logger.debug(f"Best move in this position is an attack from {src} to {tgt}.")
                return BattleCommand(src, tgt)

        self.logger.debug(f"There is no improving move in this position, ending my turn.")
        return EndTurnCommand()

    def get_best_move(self, board: Board, transfers: int) -> Tuple[int, int]:
        transfers_mutable = [transfers]
        eval, move = self.maxn(board, transfers_mutable, self.MAXN_MAX_DEPTH)
        self.logger.debug(f"Found best move: eval={eval}")
        return move

    def maxn(self, board: Board, transfers: List[int], depth: int) -> Tuple[List[float], Tuple[int, int]]:
        moves = list(possible_attacks(board, self.player_name))

        if transfers[0] < self.max_transfers: # Consider transfers only if we are still allowed to do them
            moves = moves + list(self.possible_transfers(board))

        # TODO: We have to check if the game is over when simulating, I think, or maybe not, it seems it's working like this

        if not depth or not moves:  # We reached max depth or ran out of possible moves, return just the evaluation
            return self.evaluate_state_nn(board), None

        evaluation = self.evaluate_state_nn(board)

        move = None
        new_evaluation = evaluation

        for src, tgt in moves:
            if tgt.get_owner_name() != self.player_name:
                if probability_of_successful_attack(board, src.get_name(), tgt.get_name()) < 0.5:
                    continue

            new_board = deepcopy(board)
            new_transfers = transfers

            self.simulate_move(new_board, src.get_name(), tgt.get_name(), new_transfers)

            new_evaluation, _ = self.maxn(new_board, new_transfers, depth - 1)

            index = self.players_order.index(self.player_name)
            if new_evaluation[index] > evaluation[index]:
                move = (src.get_name(), tgt.get_name())

        return new_evaluation, move

    def simulate_move(self, board: Board, src_name: int, tgt_name: int, transfers: List[int]) -> None:
        src = board.get_area(src_name)
        tgt = board.get_area(tgt_name)

        if tgt.get_owner_name() == self.player_name:  # Simulating a transfer
            transfers[0] += 1
            amount = tgt.get_dice() + src.get_dice() - 1

            if amount >= 8:
                tgt.set_dice(8)
                src.set_dice(amount - 7)
            else:
                tgt.set_dice(amount)
                src.set_dice(1)

        else:  # Simulating an attack
            tgt.set_dice(src.get_dice() - 1)
            src.set_dice(1)
            tgt.set_owner(src.get_owner_name())

    def evaluate_state(self, board: Board) -> List[float]:
        all_dices_in_game = 0
        tmp_all_dices = [0, 0, 0, 0]
        probability_of_win = [0, 0, 0, 0]
        for i in self.players_order:
            tmp_board = board.get_player_areas(i)

            for all_areas in tmp_board:
                tmp_all_dices[i-1] += all_areas.get_dice()
            all_dices_in_game += tmp_all_dices[i-1]

        for i in self.players_order:
            probability_of_win[i-1] = tmp_all_dices[i-1]/all_dices_in_game

        return probability_of_win

    def evaluate_state_nn(self, board: Board) -> List[float]:
        state = serialize_board_full(board, self.player_name)
        state = torch.FloatTensor([state])
        output = self.model(state)
        return output.tolist()[0]

    def possible_transfers(self, board: Board) -> Tuple[Area, Area]:
        for area in board.get_player_areas(self.player_name):
            if not area.can_attack():  # can_attack means that area has more than one dice => can_attack <=> can_transfer
                continue

            neighbours = area.get_adjacent_areas_names()

            for adj in neighbours:
                adjacent_area = board.get_area(adj)
                if adjacent_area.get_owner_name() == self.player_name and adjacent_area.get_dice() != 8:
                    yield (area, adjacent_area)
