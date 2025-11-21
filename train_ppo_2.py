from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from BestModelCallback import BestModelCallback
from config import PPO_TIMESTEPS, PPO_MODEL_PATH, PPO_TB_DIR
from make_env_final import make_env

def main():
    env = make_env(gui=False, normalize=True)
    logger = configure(str(PPO_TB_DIR), ["stdout", "csv", "tensorboard"])
    best_model_path = PPO_MODEL_PATH.parent / f"{PPO_MODEL_PATH.name}_best"
    callback = BestModelCallback(
        check_freq=10000,
        save_path=str(best_model_path),
        metric_name="rollout/ep_rew_mean",
        verbose=1,
    )
    model = PPO(
        policy=MlpPolicy,
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )
    model.set_logger(logger)
    model.learn(
        total_timesteps=PPO_TIMESTEPS,
        callback=callback,
    )
    model.save(str(PPO_MODEL_PATH))
    if isinstance(env, VecNormalize):
        env.save("models/ppo_hover_aviary_env.pkl")
    env.close()

if __name__ == "__main__":
    main()