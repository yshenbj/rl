from gymnasium.envs.registration import register

register(
     id="gridworld/GridWorld-v0",
     entry_point="gridworld.envs:GridWorldEnv",
     max_episode_steps=300,
)