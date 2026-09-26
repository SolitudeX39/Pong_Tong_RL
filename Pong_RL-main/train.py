import ale_py
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)
from agent import Agent


class CropObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(160, w, c), dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return obs[34:194, :, :]


def make_env():
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = CropObservation(env)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    return env


if __name__ == "__main__":
    episodes = 1500
    max_episode_steps = 10000

    hidden_layer = 512
    learning_rate = 0.0001
    gamma = 0.99
    batch_size = 32
    epsilon = 1.0
    min_epsilon = 0.02
    epsilon_decay = 0.995
    target_update_freq = 1000
    min_buffer_size = 10000

    env = make_env()

    agent = Agent(
        env, hidden_layer=hidden_layer, learning_rate=learning_rate, gamma=gamma
    )

    summary_writer_suffix = f"nature_dqn_lr={learning_rate}_bs={batch_size}"

    agent.train(
        episodes=episodes,
        max_episode_steps=max_episode_steps,
        summary_writer_suffix=summary_writer_suffix,
        batch_size=batch_size,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        target_update_freq=target_update_freq,
        min_buffer_size=min_buffer_size,
    )
