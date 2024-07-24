from gymnasium.envs.registration import register

register(
    id='games/GridWorld',
    entry_point='games.envs:GridWorldEnv',
)