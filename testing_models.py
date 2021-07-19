import gym
import torch as T
from policy_network import Policy
import time

env = gym.make("CartPole-v0")

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]

agent = Policy(obs_dim=NUM_STATES, act_dim=NUM_ACTIONS)

agent.load_state_dict(T.load('model/ga_agent.pt'))
    
def ga_test(agent):
    total_rews = 0
    for _ in range(100):
        state = env.reset()
        state = T.from_numpy(state)
        ep_rews = 0
        while True:
            act_probs = agent(state.float())
            action = T.argmax(act_probs)

            state, reward, done, info = env.step(action.item())
            state = T.from_numpy(state)
            ep_rews+=reward
            #print(ep_rews)
            if done:
                break
        total_rews+=ep_rews
        
    return total_rews

if __name__=='__main__':
    score = ga_test(agent)
    print(f'Evolved agent scored: {score}')
