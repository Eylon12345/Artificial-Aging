from musepy.corpus import StateTask
import numpy as np


class Curriculum(StateTask):
    RANDOM_TYPE = "random"        # picks a random task from the curriculum and evaluates performance
    THRESHOLD_TYPE = "threshold"  # incrementally increase difficultly by selecting "next" tasks if threshold is reached

    def __init__(self, target_magnitude=2.5, **kwargs):
        super(Curriculum, self).__init__(**kwargs)
        self.target_magnitude=2.5

    def evaluate(self, env, **kwargs):
        state_reward, done, info = StateTask.evaluate(self, env, **kwargs)

        # ADAMS TASK
        # ...
        state = env.state.get_val(channel=env.state.TYPE_CHL)

        # new target ground level
        grounded_state = state - np.sign(state) * self.target_magnitude
        magnitude_penalty = grounded_state**2
        info["magnitude_penalty"] = magnitude_penalty
        reward = state_reward - magnitude_penalty.mean()

        shape = env.shape
        right_upright_reward = np.mean((grounded_state[:, shape[1]//2:, 0] > 0) + (grounded_state[:, shape[1]//2:, 0] < 0))
        left_upright_reward = np.mean((grounded_state[:, shape[1]//2:, 1] < 0) + (grounded_state[:, shape[1]//2:, 1] > 0))
        info["right_upright_reward"] = right_upright_reward
        info["left_upright_reward"] = left_upright_reward

        reward += right_upright_reward + left_upright_reward

        return reward, done, info
