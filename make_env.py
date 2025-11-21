import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import ENV_ID, SEED, PPO_LOG_DIR
from fix_obs_wrapper import FixObsShape

def make_env(gui=False, normalize=True):
    def _init():
        env = gym.make(ENV_ID, gui=gui)
        env.reset(seed=SEED)
        env.action_space.seed(SEED)

        env = FixObsShape(env)
        env = Monitor(env, str(PPO_LOG_DIR))

        return env

    vec_env = DummyVecEnv([_init])

    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    return vec_env