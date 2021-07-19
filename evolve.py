import numpy as np
import pandas as pd
import gym
from evolution_functions import evaluate_agents, crossover, mutate, selection
import torch.nn as nn
import torch as T

from policy_network import Policy

EPOCHS = 100
POP_SIZE = 20
MUTATION_RATE = 0.2
ELITISM_RATE = 0.1

env = gym.make("CartPole-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]

if __name__=='__main__':
    #initialize population
    agents = []
    for _ in range(POP_SIZE):
        agent = Policy(obs_dim=NUM_STATES, act_dim=NUM_ACTIONS)
        agents.append(agent)
    
    fitness_list = evaluate_agents(env, agents)
    
    population = pd.DataFrame({'agents': agents,
                               'fitness': fitness_list})
    
    # Sort population dataframe descending by fitness (highest fitness at row 0)
    population = population.sort_values('fitness', ascending=False, ignore_index=True)
    
    for epoch in range(EPOCHS):
        evolved_agents = [] #new population of agents to be populated
        # Perform Elitism
        for i in range(int(POP_SIZE*ELITISM_RATE)):
            evolved_agents.append(population.iloc[i]['agents'])
        
        while len(evolved_agents)<POP_SIZE:
            # Perform Selection
            parent_a, parent_b = selection(population)
            
            # Perform Crossover
            offspring_a, offspring_b = crossover(parent_a, parent_b)

            # Perform Mutation only on offspring a (arbitrary choice)
            #offspring_a = mutate(offspring_a, MUTATION_RATE)
            
            # Add to new agents population
            evolved_agents.append(offspring_a)
            evolved_agents.append(offspring_b)
        
        population['agents'] = evolved_agents
        fitness_list = evaluate_agents(env, evolved_agents)
        
        population['fitness'] = fitness_list
        
        population = population.sort_values('fitness', ascending=False, ignore_index=True)
        best_fit = population.iloc[0]['fitness']
        print(f'epoch: {epoch}, current best: {best_fit}')
    best_agent = population.iloc[0]['agents']
    

        
            
            
    