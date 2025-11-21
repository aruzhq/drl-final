import time
import numpy as np
import gymnasium as gym
import gym_pybullet_drones
from gym_pybullet_drones.utils.enums import ActionType

def main():
    env = gym.make(
        "hover-aviary-v0",
        gui=True,
        act=ActionType.PID
    )

    obs, _ = env.reset()
    hz = 60
    dt = 1/hz

    base = np.array([0.0, 0.0, 1], dtype=np.float32)  
    radius = 0.02  
    speed = 0.5   

    start = time.time()
    duration = 20

    while time.time() - start < duration:
        t = time.time() - start

        x = base[0] + radius * np.cos(t * speed)
        y = base[1] + radius * np.sin(t * speed)
        z = base[2]

        action = np.array([[x, y, z]], dtype=np.float32)

        obs, reward, done, trunc, info = env.step(action)
        time.sleep(dt)

    env.close()

if __name__ == "__main__":
    main()