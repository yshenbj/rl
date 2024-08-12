from gymnasium.envs.registration import register

register(
    id = 'games/GridWorld',
    entry_point = 'games.envs:GridWorldEnv'
)

register(
    id = 'games/Blackjack',
    entry_point = 'games.envs:BlackjackEnv'
)

register(
    id = 'games/TicTacToe',
    entry_point = 'games.envs:TicTacToeEnv'
)

register(
    id = 'games/Gomoku',
    entry_point = 'games.envs:GomokuEnv'    
)