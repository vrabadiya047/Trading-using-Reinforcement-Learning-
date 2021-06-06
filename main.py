

# Gym, stable baseline and processing libraries
import pip

!pip install tensorflow-gpu==1.15.0 gym_anytrading gym tensorflow==1.15.0 stable_baselines
import gym
import gym_anytrading

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv
# Data bring from csv file

df = pd.read_csv('gme.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Build Environment

environment = gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)

print(environment.action_space)

state = environment.reset()
while True:
    action = environment.action_space.sample()
    n_state, reward, done, info = environment.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(15, 5))
plt.cla()
environment.render_all()
plt.show()

# Build environment and train

env_mkr = lambda: gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
env = DummyVecEnv([env_mkr])

model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Evalution

env = gym.make('stocks-v0', df=df, frame_bound=(90,110), window_size=5)
obs = env.reset()
while True:
    obs = obs[np.newaxis,...]
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        print("info", info)
        break


plt.figure(figsize=(15, 5))
plt.cla()
environment.render_all()
plt.show()
