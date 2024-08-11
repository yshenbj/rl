import torch
import time
import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter

config_dict = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'run_name': "breakout-experiment-1", # Name of the experiment/run for logging
    #'gym_id': "BreakoutNoFrameskip-v4",  # Name of Atari Gym environment
    'gym_id': "FreewayNoFrameskip-v4",

    'total_timesteps': 10000000,   # Total number of training steps
    'n_envs': 8,                   # Total number of vectorized environments
    'n_steps': 128,                # Number of steps performed in each environment for each rollout
    'n_minibatches': 4,            # Number of minibatches training batch is split into
    'update_epochs': 4,            # Number of full learning steps
    'frame_skip': 1,               # Number of frames to skip in Atari environment
    'hidden_size': 64,             # Number of neurons in actor and critic network hidden layers
    'learning_rate': 2.5e-4,       # Optimizer learning rate
    'anneal_lr': True,             # Toggle learning rate annealing
    'gamma': 0.99,                 # Discount factor
    'gae': True,                   # Toggle general advantage estimation
    'gae_lambda': 0.95,            # Lambda value used in gae calculation
    'clip_coef': 0.1,              # Amount of policy clipping
    'norm_advantages': True,       # Toggle advantage normalization
    'clip_value_loss': True,       # Toggle value loss clipping
    'weight_value_loss': 0.5,      # Weight of value loss relative to policy loss
    'weight_ent_loss': 0.01,       # Entropy loss weight
    'max_grad_norm': 0.5           # Global gradient clipping max norm
}

config_dict['batch_size'] = int(config_dict['n_envs'] * config_dict['n_steps'])
config_dict['minibatch_size'] = int(config_dict['batch_size'] // config_dict['n_minibatches'])

# Convert to a struct esque structure
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

config = Config(config_dict)

run_name = config.run_name
"""
writer = SummaryWriter(f"logs/{run_name}")

# Record hyperparameter settings
writer.add_text(
    "Hyperparameters",
    "| Param | Value |\n| ----- | ----- |\n%s" % ("\n".join([f"| {key} | {value} |" for key, value in config_dict.items()]))
)
"""

# This wrapper is an exact copy of the SB3 wrapper
class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    Args:
        env (gym.Env): The environment to wrap
    """
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info
    
    # This wrapper is an exact copy of the SB3 wrapper
class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.
    Args:
        env (gym.Env): The environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)
    
    def reward(self, reward: float) -> float:
        return np.sign(reward)

def AtariWrappers(env):
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,                   # 30 random actions a the beginning of an episode
        frame_skip=config.frame_skip,  # Repeats each input 3 times
        screen_size=84,                # Changes observation size to 84x84
        terminal_on_life_loss=True,    # Returns done=True if episode terminates
        grayscale_obs=True,            # Convert RGB to grayscale
        scale_obs=True,                # Scales observations to range 0-1
    )
    return env

def make_env(gym_id, idx, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = AtariWrappers(env)                           # Use preconfigured Atari preprocessing wrapper
        if 'FIRE' in env.unwrapped.get_action_meanings():  # Automatically 'fire' at the start
            env = FireResetEnv(env)                        
        env = ClipRewardEnv(env)                           # Clip all rewards to {-1, 0, +1}
        env = gym.wrappers.FrameStack(env, 4)              # Use stacks of 4 frames for each observation
        return env
    return thunk

envs = gym.vector.SyncVectorEnv([
    make_env(config.gym_id,
              i, 
              config.run_name) for i in range(config.n_envs)
    ])


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class PPONetwork(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=512):
        super().__init__()

        self.base = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size, n_actions), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 1), std=1.)
        )

    def forward(self, x, action=None):
        """
        Returns:
            action (torch.tensor): Action predicted by agent for each state in the batch
            log_probs (torch.tensor): The log probability of the agent taking an action following the current policy
            entropy (torch.tensor): The entropy of each distribution in the batch
            value (torch.tensor): The predicted value of each state in the batch using the critic network
        """
        x = self.base(x)
        probs = torch.distributions.Categorical(logits=self.actor(x))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def forward_critic(self, x):
        x = self.base(x)
        return self.critic(x)
    
    def select_action(self, x):
        x = self.base(x)
        probs = torch.distributions.Categorical(logits=self.actor(x))
        return probs.sample()

    def select_action_deterministic(self, x):
        x = self.base(x)
        logits = self.actor(x)
        return torch.argmax(logits, dim=-1)
    
input_shape = envs.single_observation_space.shape
n_actions = envs.single_action_space.n

network = PPONetwork(input_shape, n_actions, config.hidden_size).to(config.device)

# Change Adam epsilon to 1e-5 for increased stability
optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, eps=1e-5)

states = torch.zeros((config.n_steps, config.n_envs) + envs.single_observation_space.shape).to(config.device)
actions = torch.zeros((config.n_steps, config.n_envs)).to(config.device)
rewards = torch.zeros((config.n_steps, config.n_envs)).to(config.device)
dones = torch.zeros((config.n_steps, config.n_envs)).to(config.device)
logprobs = torch.zeros((config.n_steps, config.n_envs)).to(config.device)
values = torch.zeros((config.n_steps, config.n_envs)).to(config.device)

# Logging and training information
global_step = 0
start_time = time.time()
num_updates = config.total_timesteps // config.batch_size # Total number of learning update that will be performed

# Initial state and done flag
state = torch.Tensor(envs.reset()[0]).to(config.device)
done = torch.zeros(config.n_envs).to(config.device)

# Manually initialize metrics for tracking as RecordEpisodeStatistics is currently broken for vectorized environments
episodic_return = np.zeros([config.n_envs])
episode_step_count = np.zeros([config.n_envs])

for update in range(1, num_updates + 1):
    # Annealed learning rate
    if config.anneal_lr:
        fraction = 1.0 - ((update - 1.0) / num_updates)
        lr_current = fraction * config.learning_rate
        optimizer.param_groups[0]['lr'] = lr_current
    
    # Rollouts
    for step in range(0, config.n_steps):
        global_step += 1 * config.n_envs
        states[step] = state
        dones[step] = done

        # Action selection doesn't require gradient updates
        with torch.no_grad():
            action, logprob, _, value = network(state)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        state, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        rewards[step] = torch.tensor(reward).to(config.device).view(-1)
        state = torch.Tensor(state).to(config.device)
        done = torch.Tensor(done).to(config.device)

        # Logging
        episodic_return += reward
        episode_step_count += 1

        # If an episode is done
        if 'final_observation' in info.keys():
            for i_env, done_flag in enumerate(info['_final_observation']):
                if done_flag:
                    print("\rglobal_step={global_step}, episodic_return={episodic_return[i_env]}", end='')

                    #writer.add_scalar('charts/episodic_return', episodic_return[i_env], global_step)
                    #writer.add_scalar('charts/episodic_length', episode_step_count[i_env], global_step)

                    episodic_return[i_env], episode_step_count[i_env] = 0., 0.
    # General advantage estimation
    with torch.no_grad():
        next_value = network.forward_critic(state).reshape(1, -1)
        if config.gae:
            advantages = torch.zeros_like(rewards).to(config.device)
            lastgaelam = 0
            for t in reversed(range(config.n_steps)):
                if t == config.n_steps - 1:
                    next_non_terminal = 1.0 - done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + config.gamma * next_values * next_non_terminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * next_non_terminal * lastgaelam
            returns = advantages + values
        # Vanilla advantage calculation 
        else:
            returns = torch.zeros_like(rewards).to(config.device)
            for t in reversed(range(config.num_steps)):
                if t == config.n_steps - 1:
                    next_non_terminal = 1.0 - done
                    next_return = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + config.gamma * next_non_terminal * next_return
            advantages = returns - values

    # Tracking metric
    clipfracs = []

    # Minibatch update
    # Flatten batches b_ indicates a full batch, mb_ indicates a minibatch
    b_states = states.reshape((-1,) + input_shape)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_logprobs= logprobs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    
    # Optimizing the policy and value network
    b_inds = np.arange(config.batch_size) # Batch indices

    for epoch in range(config.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, config.batch_size, config.minibatch_size):
            end = start + config.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = network(b_states[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            # Tracking metrics
            with torch.no_grad():
                # Calculate approx KL divergence between policies http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - logratio).mean()
                # Number of triggered clips
                clipfracs +=  [((ratio - 1.).abs() > config.clip_coef).float().mean()]

            # Advantage normalization
            mb_advantages = b_advantages[mb_inds]
            if config.norm_advantages:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            # Clipped surrogate objective
            loss_surrogate_unclipped = -mb_advantages * ratio
            loss_surrogate_clipped = -mb_advantages * torch.clip(ratio, 
                                             1 - config.clip_coef, 
                                             1 + config.clip_coef)
            loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

            # Value loss clipping
            newvalue = newvalue.view(-1)
            if config.clip_value_loss:
                loss_v_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                value_clipped = b_values[mb_inds] + torch.clip(
                    newvalue - b_values[mb_inds],
                    -config.clip_coef,
                    config.clip_coef
                )
                loss_v_clipped = (value_clipped - b_returns[mb_inds]) ** 2 # MSE
                loss_v_max = torch.max(loss_v_unclipped, loss_v_clipped)
                loss_value = 0.5 * loss_v_max.mean()
            else:
                loss_value = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() # MSE
            
            # Entropy loss
            loss_entropy = entropy.mean()
            # Weighted value loss
            loss = loss_policy + config.weight_ent_loss * -loss_entropy + config.weight_value_loss * loss_value

            # Global gradient clipping
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    
    # Number of steps per second
    sps = int(global_step / (time.time() - start_time))
    mean_clipfracs = np.mean([item.cpu().numpy() for item in clipfracs])