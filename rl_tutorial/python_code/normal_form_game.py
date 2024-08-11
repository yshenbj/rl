import matplotlib.colors as colours
from mdp import *
import matplotlib.pyplot as plt
import numpy as np

from rendering_utils import COLOURS


class NormalFormGame:

    """Create a two-player normal form game from a list of players, a map of actions for each player, and a matrix of rewards.
    'players' is a list of strings of agent IDs
    'actions' is a list of tuples of actions: one list for each agent
    'rewards' is a dictionary mapping each pair of actions with an outcome
    """

    def __init__(
        self, player_1_name, player_2_name, player_1_actions, player_2_actions, rewards
    ):
        self.player_1_name = player_1_name
        self.player_2_name = player_2_name
        self.player_1_actions = player_1_actions
        self.player_2_actions = player_2_actions
        self.rewards = rewards

    """ Get the list of players/players for this game as a list [1, ..., N] """

    def get_players(self):
        return [self.player_1_name, self.player_2_name]

    """ Return the reward/payoff for set of actions """

    def get_reward(self, action_1, action_2):
        return self.rewards[(action_1, action_2)]

    def visualise(self, grid_size=1.5, gif=False):
        """
        Visualizes a two-player normal form game using Matplotlib.
        """
        num_rows, num_cols = len(self.player_1_actions), len(self.player_2_actions)

        # Plot the payoff matrices

        fig, ax = plt.subplots(
            1, 1, figsize=(num_cols * grid_size, num_rows * grid_size)
        )

        # Player 1's payoff matrix
        img = [[COLOURS["white"] for _ in range(num_cols)] for _ in range(num_rows)]
        im1 = ax.imshow(img)

        # Add gridlines between cells
        for i in range(num_rows + 1):
            ax.axhline(i - 0.5, color="grey", lw=0.5)
        for j in range(num_cols + 1):
            ax.axvline(j - 0.5, color="grey", lw=0.5)

        # Display payoffs in cells
        for y in range(num_cols):
            for x in range(num_rows):
                plt.text(
                    y,
                    x,
                    self.rewards[(self.player_1_actions[x], self.player_2_actions[y])],
                    fontsize="large",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        # Set ticks and labels
        ax.set_xticks(np.arange(num_cols))
        ax.set_yticks(np.arange(num_rows))
        ax.set_xticklabels(
            self.player_2_actions,
            fontsize="large",
        )
        ax.set_yticklabels(
            self.player_1_actions,
            fontsize="large",
        )
        ax.set_title(
            self.player_2_name,
            fontsize="x-large",
        )

        max_y_tick = 0
        for label in self.player_1_actions:
            if len(label) > max_y_tick:
                max_y_tick = len(label)

        player_1_name_x = -0.5 - (
            max_y_tick * 0.15
        )  # These parameters were adjusted experimentally
        player_1_name_y = (num_rows - 1) / 2  # Centre the player name vertically
        plt.text(
            player_1_name_x,
            player_1_name_y,
            self.player_1_name,
            fontsize="x-large",
            horizontalalignment="center",
            verticalalignment="center",
            rotation=90,
        )
        ax.tick_params(
            which="both", top=True, left=False, right=False, bottom=False, length=0
        )  # Set length=0 to remove small ticks

        # Move xticks to the top
        ax.xaxis.tick_top()

        plt.setp(ax.get_xticklabels(), ha="center")

        # Show plot
        ##plt.tight_layout()

        if gif:
            return fig, ax, im
        else:
            return fig