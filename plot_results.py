from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from config import (
    PPO_LOG_DIR,
    PPO_TB_DIR,
    RESULTS_DIR,
)

def plot_rewards(log_dir: Path, algo_name: str):
    try:
        x, y = ts2xy(load_results(str(log_dir)), "timesteps")
    except Exception:
        return
    if len(x) == 0:
        return
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title(f"{algo_name.upper()} reward vs timesteps")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{algo_name}_reward.png")
    plt.close()

def plot_losses(tb_dir: Path, algo_name: str):
    progress_files = list(tb_dir.glob("**/progress.csv"))
    if not progress_files:
        return
    progress_path = progress_files[-1]
    df = pd.read_csv(progress_path)
    if "time/total_timesteps" in df.columns:
        x = df["time/total_timesteps"]
    else:
        return
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    if not loss_cols:
        return
    plt.figure()
    for col in loss_cols:
        plt.plot(x, df[col], label=col)
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.title(f"{algo_name.upper()} losses vs timesteps")
    plt.legend()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{algo_name}_loss.png")
    plt.close()

def main():
    algo_configs = [
        ("ppo", PPO_LOG_DIR, PPO_TB_DIR),
    ]
    for name, log_dir, tb_dir in algo_configs:
        plot_rewards(log_dir, name)
        plot_losses(tb_dir, name)

if __name__ == "__main__":
    main()