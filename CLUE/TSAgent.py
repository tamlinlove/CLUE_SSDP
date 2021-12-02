import numpy as np
from itertools import product

from Agent import Agent
from Utility import Utility
from StateTable import StateTable

class TSAgent(Agent):
    '''
    Class for a Thompson Sampling Agent
    Uses a Gaussian model of each action's rewards for each state
    '''
    def __init__(self,env,alpha=1,beta=10,mu_0=3):
        '''
        Initialise action-value epsilon-greedy SSDP agent

        Input:
            env - instance of InfluenceDiagram class representing the environment
            alpha - gamma shape parameter
            beta - gamma rate parameter
            mu_0 - prior mean
        '''
        self.name = "UCB Baseline Agent"
        # Initialise state and action space
        self.state_space = env.state_space
        self.action_space = env.action_space

        # Initialise dictionary of node types
        self.node_types = env.node_types
        self.domains = env.domains

        # Parameters
        self.alpha_0 = alpha
        self.beta_0 = beta
        self.mu_0 = mu_0

    def act(self,state):
        '''
        Select an action given a state

        Input:
            state - dict mapping state variable names to values
        Output:
            action - dict mapping action variable names to values
        '''
        samples,indices = self.sample(state)
        vals = np.asarray(samples)
        best_index = indices[np.argmax(np.random.random(vals.shape) * (vals==vals.max()))]
        best_action = self.rewards.index_to_assignment(best_index)
        return best_action


    def learn(self,state,actions,reward):
        '''
        Update action-value function based on state-action-reward

        Input:
            state - a dict mapping state node names to values
            actions - a dict mapping action node names to values
            reward - the reward of this trial
        '''
        assignment = {}
        for node in state:
            assignment[node] = state[node]
        for node in actions:
            assignment[node] = actions[node]
        v = self.visits.get_value(assignment)
        alpha = self.alphas.get_value(assignment) + 0.5
        self.alphas.add_to_table(assignment,alpha)
        beta = self.betas.get_value(assignment) + (v/(v+1))*(((reward-self.mu_0)**2)/2)
        self.betas.add_to_table(assignment,beta)

        v_0 = beta/(alpha+1)
        rewards = self.rewards.get_value(assignment).copy()
        rewards.append(reward)
        self.rewards.add_to_table(assignment,rewards)
        self.visits.add_to_table(assignment,v+1)
        self.mus.add_to_table(assignment,np.mean(rewards))


    def reset(self):
        self.alphas = StateTable(self.domains,default_value=self.alpha_0)
        self.betas = StateTable(self.domains,default_value=self.beta_0)
        self.rewards = StateTable(self.domains,default_value=[])
        self.mus = StateTable(self.domains,default_value=self.mu_0)
        self.visits = StateTable(self.domains,default_value=0) # Reset visit counts

    def sample(self,state):
        samples = []
        combinations = list(product(*self.action_space.values())) # All possible actions
        keys = list(self.action_space.keys())
        indices = []
        for action in combinations:
            assignment = state.copy()
            for i in range(len(keys)):
                assignment[keys[i]] = action[i]
            indices.append(self.rewards.assignment_to_index(assignment))
            tau = np.random.gamma(self.alphas.get_value(assignment),1/self.betas.get_value(assignment))
            if tau == 0 or self.visits.get_value(assignment) == 0:
                tau = 0.001
            var = 1/tau
            samples.append(np.random.normal(self.mus.get_value(assignment), np.sqrt(var)))
        return samples,indices
