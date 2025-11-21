from pathlib import Path

ENV_ID = "hover-aviary-v0"
SEED = 0

PPO_TIMESTEPS = 500_000

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"
TB_DIR = BASE_DIR / "tensorboard"
RESULTS_DIR = BASE_DIR / "results"

PPO_LOG_DIR = LOG_DIR / "ppo"
PPO_TB_DIR = TB_DIR / "ppo"
PPO_MODEL_PATH = MODEL_DIR / "ppo_hover_final"

for path in [
    LOG_DIR,
    MODEL_DIR,
    TB_DIR,
    RESULTS_DIR,
    PPO_LOG_DIR,
    PPO_TB_DIR
]:
    path.mkdir(parents=True, exist_ok=True)
