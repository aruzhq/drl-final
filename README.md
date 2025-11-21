# **README.md**

# Drone Hover Control using Reinforcement Learning (PPO)

This project implements a reinforcement-learning pipeline for autonomous drone stabilization in the *hover-aviary-v0* environment from **gym-pybullet-drones**.
The system trains a Proximal Policy Optimization (PPO) agent to maintain a stable hover position in a continuous-control setting, using realistic quadrotor physics and noise models.

The project includes:

* modular environment construction,
* PPO training with Stable-Baselines3,
* automatic best-model checkpointing,
* evaluation scripts with static and dynamic targets,
* TensorBoard logging,
* offline plotting of rewards and losses.

---

## **Project Structure**

```
FINAL PROJECT/
├─ logs/
│  └─ ppo/                      # Monitor logs (monitor.csv)
├─ models/                      # Saved PPO models
├─ results/                     # Reward/loss plots
├─ tensorboard/
│  └─ ppo/                      # TensorBoard event files
├─ BestModelCallback.py         # Custom checkpoint callback
├─ config.py                    # Global configuration
├─ make_env.py                  # Environment factory
├─ train_ppo_2.py               # Training script
├─ evaluate_1.py                # Static-hover evaluation
├─ evaluate_2.py                # Circular trajectory eval
├─ plot_results.py              # Reward/loss plot generator
└─ requirements.txt
```

---

## **Installation**

### **1. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate        # Windows
```

### **2. Install dependencies**

The project uses a minimal requirements list to avoid platform-specific pinning:

```
pybullet
gym
stable-baselines3
gym-pybullet-drones @ git+https://github.com/utiasDSL/gym-pybullet-drones.git
matplotlib
```

Install via:

```bash
pip install -r requirements.txt
```

### **Optional recommended libraries (present in development environment)**

```
numpy
pandas
torch
tensorboard
scipy
```

These are automatically installed when Stable-Baselines3, PyBullet, and TensorBoard are installed, but can be listed explicitly if required.

---

## **Training**

Run PPO training for 500,000 timesteps:

```bash
python train_ppo_2.py
```

Logs are written to:

* `logs/ppo/monitor.csv` — Episode rewards/lengths
* `tensorboard/ppo/` — Detailed training metrics
* `models/` — Final and best model checkpoints

TensorBoard can be launched via:

```bash
tensorboard --logdir tensorboard/
```

---

## **Evaluation**

### **Static hovering**

```bash
python evaluate_1.py
```

### **Circular trajectory tracking**

```bash
python evaluate_2.py
```

---

## **Plotting Results**

Generate reward and loss curves from logs:

```bash
python plot_results.py
```

Plots are saved in:

```
results/
 ├─ ppo_reward.png
 └─ ppo_loss.png
```

---

## **Environment**

The project was developed under Python 3.10 with the following key libraries:

* `stable_baselines3==2.7.0`
* `torch==2.9.1`
* `gymnasium==1.2.2`
* `gym-pybullet-drones==2.0.0`
* `pybullet==3.2.7`
* `matplotlib==3.10.7`
* `pandas==2.3.3`
* `tensorboard==2.20.0`

The full development environment is shown in the pip list above.
