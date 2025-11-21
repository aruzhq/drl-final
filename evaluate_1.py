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

    target = np.array([[0, 0, 1]], dtype=np.float32)  
    
    start = time.time()
    duration = 20 

    while time.time() - start < duration:
        action = target.copy()

        obs, reward, done, trunc, info = env.step(action)
        time.sleep(dt)

    env.close()

if __name__ == "__main__":
    main()