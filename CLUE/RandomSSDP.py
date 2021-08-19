'''
A random SSDP, as defined by an influence diagram
'''

from InfluenceDiagram import InfluenceDiagram
import PGM

import numpy as np
import random

class RandomSSDP(InfluenceDiagram):
    '''
    The class for a random SSDP influence diagram
    '''
    def build_network(self,num_chance=7,num_decision=3,seed=1,reward_range=[-1,1]):
        '''
        Builds a random SSDP influence diagram
        This is called by the initialisation of the InfluenceDiagram class

        Usage:
            env.build_network(num_chance,num_decision,seed)

        Input:
            num_chance - the number of boolean chance variables
            num_decision - the number of boolean decision variables
            seed - the random seed used to generate the environment
            reward_range - a list [min,max], where min is the minimum possible reward, and max is the maximum
                default - [-1,1]
        '''
        # Name environment
        self.name = "Random ("+str(num_chance)+","+str(num_decision)+")"

        # Seed
        random.seed(seed)
        np.random.seed(seed)

        # State and Action Spaces
        boolean = [False,True]
        self.state_space = {} # Maps state node name to value
        self.action_space = {} # Maps action node name to value
        for i in range(num_chance):
            self.state_space["C"+str(i)] = boolean # S = {C1,C2,...}
        for i in range(num_decision):
            self.action_space["A"+str(i)] = boolean # A = {A1,A2,...}

        # State Variables
        self.variables = {} # Maps node name to variable object
        chance_vars = []
        for node in self.state_space:
            self.variables[node] = PGM.Variable(node,self.state_space[node]) # Create variable
            chance_vars.append(self.variables[node])

        # Action Variables
        action_vars = []
        self.parents = {}
        curr_dec = []
        '''
        Each decision node is parent to all subsequent decision nodes
        This is because actions are assigned in order A1 -> A2 -> ...
        '''
        for node in self.action_space:
            self.parents[node] = list(self.state_space.keys()) + curr_dec
            curr_dec.append(node)
        for node in self.action_space:
            pars = []
            for par in self.parents[node]:
                pars.append(self.variables[par])
            self.variables[node] = PGM.DecisionVariable(node,self.action_space[node],set(pars)) # Create variable
            action_vars.append(self.variables[node])

        # Factors
        '''
        Randomly assign parents to each state node
        Ci is a potential parent to all variables Cj where j>i
        '''
        potential_parents = []
        for node in self.state_space:
            num_pars = np.random.randint(0,len(potential_parents)+1)
            self.parents[node] = np.random.choice(potential_parents,num_pars,replace=False)
            potential_parents.append(node)

        factor_dict = {}
        for node in list(self.state_space.keys()):
            num_pars = len(self.parents[node])
            num_prob_pairs = len(self.state_space[node])**num_pars
            factor = []
            for i in range(num_prob_pairs):
                p1 = round(np.random.uniform(0,1),3) # Rounding to avoid floating point errors
                factor.append(p1)
                factor.append(round(1-p1,3))
            factor_dict[node] = factor

        self.factors = {}
        factor_list = []
        for node in factor_dict:
            pars = []
            for par in self.parents[node]:
                pars.append(self.variables[par])
            self.factors[node] = PGM.Prob(self.variables[node],pars,factor_dict[node]) # Create CPD
            factor_list.append(self.factors[node])

        # Utility
        num_pars = np.random.randint(0,len(chance_vars)+1)
        '''
        Randomly assign parents to reward
        All decision variables are reward parents to ensure actions are meaningful
        '''
        reward_parents = list(np.random.choice(list(self.state_space.keys()),num_pars,replace=False))+list(self.action_space.keys())
        self.parents["reward"] = reward_parents
        reward_parent_vars = [self.variables[par] for par in reward_parents]
        utility = np.random.uniform(reward_range[0],reward_range[1],size=2**len(reward_parents))
        self.factors["reward"] = PGM.Utility(reward_parent_vars,utility) # Create utility

        self.dn = PGM.DecisionNetwork(chance_vars+action_vars,factor_list+[self.factors["reward"]]) # Create ID
        self.gm = PGM.Graphical_model(chance_vars+action_vars,factor_list) # Create GM
