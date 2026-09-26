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
    env = make_env()
    agent = Agent(env, hidden_layer=512)
    agent.test("models/model_ep1000.pt")
    # agent.test('models/latest.pt')
