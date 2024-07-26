from gymnasium.envs.registration import register

register(
    id = 'games/GridWorld',
    entry_point = 'games.envs:GridWorldEnv'
)

register(
    id = 'games/BlackJack',
    entry_point = 'games.envs:BlackjackEnv'
)