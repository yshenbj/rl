import cv2
from itertools import count
import gymnasium as gym

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mdp import MDP

"""
A wrapper class around the gymnasium class for the Arcade Learning Environment
(https://gymnasium.farama.org/environments/atari/)
to meet the requirements for the MDP class interface.
"""


class ALEWrapper(MDP):
    def __init__(self, version, render_mode="rgb_array", discount_factor=1.0):
        self.env = gym.make(version, render_mode=render_mode)
        observation, info = self.env.reset()
        self.terminated = False
        self.discount_factor = discount_factor
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def get_actions(self, state=None):
        num_actions = self.env.action_space.n
        return list(range(num_actions))

    def get_initial_state(self):
        observation, info = self.env.reset()
        return tuple(observation)

    def reset(self):
        observation, info = self.env.reset()
        return tuple(observation)

    def step(self, action):
        return self.env.step(action)

    def set_render_mode(self, render_mode):
        self.env.render_mode = render_mode

    def render(self):
        return self.env.render()

    """ Return true if and only if state is a terminal state of this MDP """

    def is_terminal(self, state):
        # This hacks the gym interface by recording termination status during 'execute'
        if self.terminated:
            self.env.reset()
            self.terminated = False
            return True
        return self.terminated

    """ Return the discount factor for this MDP """

    def get_discount_factor(self):
        return self.discount_factor

    def execute(self, state, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated or truncated
        return (tuple(observation), reward, terminated)

    """ Execute a policy on an environment and return frames from it. """
    def get_frames(self, policy, max_episode_length=float("inf")):

        # Check that the render mode is rgb_array
        assert (
            self.env.render_mode == "rgb_array"
        ), 'The MDP instance must be created with render_mode="rgb_array" in the environment\'s initialisation'

        state = self.reset()
        frames = []
        for steps in count():
            # Step using random actions
            action = policy.select_action(state, self.get_actions(state))
            next_state, reward, done = self.execute(state, action)
            state = next_state
            frames.append(self.render())
            if done or steps == max_episode_length:
                break
        
        return frames
    
    """ Execute a policy on an environment and create a video from it. """

    def create_video(self, policy, filename, max_episode_length=float("inf")):
        frames = self.get_frames(policy, max_episode_length=max_episode_length)

        out = cv2.VideoWriter(
            filename + ".mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,
            (frames[0].shape[1], frames[0].shape[0]),
        )

        for i in range(len(frames)):
            out.write(frames[i])
        out.release()

    """ Execute a policy on an environment and create a gif from it """

    def create_gif(self, policy, filename, max_episode_length=float("inf")):
        frames = self.get_frames(policy, max_episode_length=max_episode_length)
        plt.figure(figsize=(frames[0].shape[1]/30.0, frames[0].shape[0]/30.0), dpi=100)
        patch = plt.imshow(frames[0])
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        def update(frame):
            patch.set_data(frame)
            return patch,

        anim = animation.FuncAnimation(plt.gcf(), update, frames=frames, interval=50)
        anim.save(filename + ".gif")
        plt.close()
