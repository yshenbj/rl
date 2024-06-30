from potential_function import PotentialFunction

class FreewayPotentialFunction(PotentialFunction):
    # The highest value for the Y position is 177 (https://arxiv.org/abs/2109.01220)
    Y_MAX=177

    # Byte 14 contains y position of agent (https://arxiv.org/abs/2109.01220)
    Y_POSITION_INDEX=14

    def __init__(self):
        self.max_visited = 0

    def get_potential(self, state):
        y_position = state[self.Y_POSITION_INDEX]
        return (y_position / self.Y_MAX)