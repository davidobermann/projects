import random
from typing import Tuple

from .discrete_space import DiscreteSpace


class SimplifiedWumpusWorld:
    """
    Worlds are described by a grid like the following

        00P0
        000P
        PW00
        00PG

    The starting point is always at (0, 0)
    0: empty tile
    P: pit, leads to game over
    W: wumpus, leads to game over
    G: gold, the agent is successful if it reaches here
    """

    def __init__(self):
        self.board = [
            ['0', '0', '|', '0', '0'],
            ['0', '0', '0', '0', '0'],
            ['|', '0', '|', '|', '0'],
            ['0', '0', '0', '0', '0'],
            ['0', '0', '|', '0', 'G'],
        ]
        self.agentX = 0
        self.agentY = 0

        self.num_actions = 4  # up, down, left, right
        self.num_spaces = len(self.board) * len(self.board[0]) # 16  # one for each tile
        self.action_space = DiscreteSpace(self.num_actions)
        self.observation_space = DiscreteSpace(self.num_spaces)

    @property
    def state(self) -> int:
        # return self.agentY * 4 + self.agentX
        return self.agentY * len(self.board[0]) + self.agentX

    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Returns a tuple in the format of (new state, reward, done)
        given an int, action, where 0 <= action < 4
        """
        assert 0 <= action < self.num_actions, "Action must be an integer between 0 and 3"
        new_agentX, new_agentY = self.agentX, self.agentY

        if action == 0:
            new_agentY = min(len(self.board) - 1, self.agentY + 1)
        elif action == 1:
            new_agentY = max(0, self.agentY - 1)
        elif action == 2:
            new_agentX = max(0, self.agentX - 1)
        else:
            new_agentX = min(len(self.board[0]) - 1, self.agentX + 1)

        if self.board[new_agentY][new_agentX] != "|":
            self.agentX, self.agentY = new_agentX, new_agentY

        if self.board[self.agentY][self.agentX] in ('P', 'W'):
            return self.state, -1000, True
        elif self.board[self.agentY][self.agentX] == 'G':
            return self.state, 1000, True
        else:
            return self.state, -1, False

    def reset(self) -> int:
        self.agentX = random.randint(0, len(self.board[0]) - 1)
        self.agentY = random.randint(0, len(self.board) - 1)
        while self.board[self.agentY][self.agentX] != '0':
            self.agentX = random.randint(0, len(self.board[0]) - 1)
            self.agentY = random.randint(0, len(self.board) - 1)

        return self.state

    @property
    def width(self) -> int:
        return len(self.board)

    @property
    def height(self) -> int:
        return len(self.board[0])
