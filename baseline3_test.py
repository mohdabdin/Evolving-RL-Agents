import gym
from stable_baselines3 import A2C

env = gym.make('CartPole-v0')
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=15000)
obs = env.reset()

cum_rews = 0
for _ in range(100):
    ep_rews = 0
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_rews+=reward
        if done:
            obs = env.reset()
            break
    cum_rews+=ep_rews
    
print(cum_rews)