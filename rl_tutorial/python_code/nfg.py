import numpy as np
import matplotlib.colors as colours
import matplotlib.pyplot as plt

from rendering_utils import COLOURS

def plot_normal_form_game(player1_name, player2_name, player1_actions, player2_actions, payoffs):
    """
    Visualizes a 2-player normal form game using Matplotlib.

    Parameters:
    - player1_name (str): Name of player 1
    - player2_name (str): Name of player 2
    - player1_actions (list or array): Labels for player 1 actions
    - player2_actions (list or array): Labels for player 2 actions
    - payoffs (dict): Payoff dictionary with keys as action pairs and values as payoffs
    """
    num_rows, num_cols = len(player1_actions), len(player2_actions)

    # Plot the payoff matrices
    grid_size=2.0
    fig, axs = plt.subplots(1, 1, figsize=(num_cols * grid_size, num_rows * grid_size)) #figsize=(5, 5))

    # Player 1's payoff matrix
    img = [[COLOURS['white'] for _ in range(num_cols)] for _ in range(num_rows)]
    im1 = axs.imshow(img)

    # Add gridlines between cells
    for i in range(num_rows + 1):
        axs.axhline(i-0.5, color='grey', lw=0.5)
    for j in range(num_cols + 1):
        axs.axvline(j-0.5, color='grey', lw=0.5)

    # Display payoffs in cells
    for y in range(num_cols):
        for x in range(num_rows):
            plt.text(
                y,
                x,  
                payoffs[(player1_actions[x], player2_actions[y])],
                fontsize="x-large",
                horizontalalignment="center",
                verticalalignment="center",
            )
            print((player1_actions[x], player2_actions[y]))
    
    # Set ticks and labels
    axs.set_xticks(np.arange(num_cols))
    axs.set_yticks(np.arange(num_rows))
    axs.set_xticklabels(player2_actions)
    axs.set_yticklabels(player1_actions)
    axs.set_title(player2_name)
    plt.text(-1.0, 0.5, player1_name, fontsize="x-large", horizontalalignment="center", verticalalignment="center", rotation=90)
    axs.tick_params(which='both', top=True, left=False, right=False, bottom=False, length=0)  # Set length=0 to remove small ticks

    # Move xticks to the top
    axs.xaxis.tick_top()

    plt.setp(axs.get_xticklabels(), ha="center")

    # Show plot
    plt.tight_layout()
    plt.show()

plot_normal_form_game("Prisoner A", "Prisoner B", ["Admit", "Deny"], ["Admit", "Deny"], {
    ("Admit", "Admit"): (-2, -2),  
    ("Admit", "Deny"): (0, -4),
    ("Deny", "Admit"): (-4, 0),
    ("Deny", "Deny"): (-1, -1),

})
