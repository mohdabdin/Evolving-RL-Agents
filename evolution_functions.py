import numpy as np
import torch as T
import random
import torch.nn as nn
import torch as T
import copy

# Loop over agents and evaluate each agents performance in the environment
def evaluate_agents(env, agents):
    fitness_list = []
    for agent in agents:
        state = env.reset()
        state = T.from_numpy(state)
        ep_rewards=0
        while True:
            act_probs = agent(state.float())
            action = T.argmax(act_probs)

            state, reward, done, _ = env.step(action.item())
            state = T.from_numpy(state)
            ep_rewards+=reward

            if done or ep_rewards==19500:
                break
        
        fitness_list.append(ep_rewards)
    
    return fitness_list
    
# Helper function to apply crossover between two matrices
def single_mat_crossover(mat_a, mat_b):
    # Save shape to reshape after flattening
    mat_shape = mat_a.shape
    
    mat_a_flat = mat_a.flatten()
    mat_b_flat = mat_b.flatten()
    
    # random crossover point for flattened matrices
    crossover_point = random.randint(1, len(mat_a_flat))
    
    offspring1 = T.cat((mat_a_flat[:crossover_point], mat_b_flat[crossover_point:]))
    offspring2 = T.cat((mat_a_flat[crossover_point:], mat_b_flat[:crossover_point]))
        
    # Reshape to original matrix shape
    offspring1 = offspring1.reshape(mat_shape)
    offspring2 = offspring2.reshape(mat_shape)
    
    return offspring1, offspring2

# Apply crossover between two parent agents
def crossover(parent_a, parent_b):
    # Copy parent agents to new variables
    offspring_a = copy.deepcopy(parent_a)
    offspring_b = copy.deepcopy(parent_b)
    
    # Loop over each matrix in our two policy networks and apply crossover
    for param_a, param_b in zip(offspring_a.parameters(), offspring_a.parameters()):
        mat_a, mat_b = single_mat_crossover(param_a.data, param_b.data)
        param_a.data = nn.parameter.Parameter(T.zeros_like(param_a))
        param_b.data = nn.parameter.Parameter(mat_b)
        
    return offspring_a, offspring_b

def single_mat_mutation(mat, mutation_rate):
    mat_shape = mat.shape
    flattened_mat = mat.flatten()
    indices = random.sample(range(1, len(flattened_mat)), int(len(flattened_mat)*mutation_rate))
    
    sigma = (max(flattened_mat)-min(flattened_mat))/max(flattened_mat)
    for idx in indices:
        flattened_mat[idx] = random.gauss(flattened_mat[idx], sigma)
        
    # Reshape to original matrix shape
    mat = flattened_mat.reshape(mat_shape)
    #print(mat.shape)
    return mat

# Mutate each
def mutate(agent, mutation_rate):
    mutated_agent = copy.deepcopy(agent)
    for param in mutated_agent.parameters():
        mat = single_mat_mutation(param.data, mutation_rate)
        param.data = nn.parameter.Parameter(mat)
        
    return mutated_agent

# Select two parents from population based on fitness
def selection(population):
    # Define a probability distribution proportional to each agent's fitness/reward score
    fit_sum = sum(population['fitness'])
    prob_dist = population['fitness']/fit_sum

    # Choose random indices from distribution making sure they're not equal
    par_a_idx = 0
    par_b_idx = 0
    while par_a_idx==par_b_idx:
        par_a_idx = np.random.choice(np.arange(0, len(prob_dist)), p=prob_dist)
        par_b_idx = np.random.choice(np.arange(0, len(prob_dist)), p=prob_dist)

    parent_a = population.iloc[par_a_idx]['agents']
    parent_b = population.iloc[par_b_idx]['agents']

    # Return two picked parents
    return parent_a, parent_b
