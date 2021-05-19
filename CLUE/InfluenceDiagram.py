'''
Parent class for any SSDP influence diagram environment
'''

import numpy as np

class InfluenceDiagram():
    '''
    Parent class for any influence diagram environment

    Assumes:
        No decision variable is a parent to a chance variable

    '''
    def __init__(self,**kwargs):
        '''
        Initialises the environment

        Input:
            **kwargs - passed to build_network function
        '''
        self.name = "Untitled Environment"
        self.build_network(**kwargs) # Call the child's initialisation function

        # Create node lists
        self.chance = list(self.state_space.keys()) # Chance variables
        self.decision = list(self.action_space.keys()) # Decision variables
        self.nodes = self.chance+self.decision # All nodes

        # Initialise dictionary of node types
        self.node_types = {}
        for node in self.chance:
            self.node_types[node] = "state"
        for node in self.decision:
            self.node_types[node] = "action"

        # Initialise domains
        self.domains = {}
        for node in self.chance:
            self.domains[node] = self.state_space[node]
        for node in self.action_space:
            self.domains[node] = self.action_space[node]

        # Size
        self.size_S = None
        self.size_A = None
        self.size()

    def __str__(self):
        '''
        Print details of environment
        '''
        text = "====="+self.name+"=====\n"
        text += "State Variables : "+str(len(self.chance))+" (|S|="+str(self.size_S)+")\n"
        for node in self.chance:
            text += "\t" + node + " : " + str(self.domains[node]) + "\n"
        text += "Action Variables : "+str(len(self.decision))+" (|A|="+str(self.size_A)+")\n"
        for node in self.decision:
            text += "\t" + node + " : " + str(self.domains[node]) + "\n"
        text += "Parents\n"
        for node in self.parents:
            text += "\t" + node + " : " + str(self.parents[node]) + "\n"
        text += "For CPDs and reward function, call InfluenceDiagram.print_CPD()"
        return text

    def build_network(self):
        '''
        Constructs the environment. The function must set the following variables
            self.state_space - dict mapping state variable name to domain list
            self.action_space - dict mapping action variable name to domain list
            self.factors - dict mapping node name to CPD over that node's variable
                self.factors["reward"] = the utility object
            self.variables - dict mapping node name to that node's variable
            self.parents - dict mapping node name to names of its parents
            self.dn - decison network object
            self.gm - graphical model object
        '''
        raise NotImplementedError()

    def order(self):
        '''
        Return a list of nodes in the order they are sampled
        '''
        return self.nodes

    def print_CPD(self):
        '''
        Print the CPDs and utility
        Warning: For large |S| and |A|, can get very big
        '''
        text = "====="+self.name+"=====\n"
        for node in self.factors:
            text += "===" + node + "===\n"
            text += str(self.factors[node])
        print(text)

    def reset(self):
        '''
        Resets the environments current state and samples a new trial
        '''
        self.state = {} # Reset to empty state
        return self.sample() # Sample new state

    def sample(self):
        '''
        Samples a random assignment of state variables using defined CPD

        Output:
            state - a dict mapping node name to value in node domain, represents current state
        '''
        for node in self.chance:
            distribution = self.factors[node].cond_dist(self.state_vars) # Fetch conditional distribution on parents in assignment
            value = np.random.choice(list(distribution.keys()),p=list(distribution.values())) # Chose a value with probability
            self.state[node] = value # Set variable in current state
            self.state_vars[self.variables[node]] = value # Set variable in current state
        return self.state

    def size(self):
        '''
        Returns the size of the environment

        Output:
            size_S - the cardinality of the state space
            size_A - the cardinality of the action space
        '''
        if self.size_S is None:
            self.size_S = 1
            for node in self.state_space:
                    self.size_S *= len(self.state_space[node])
        if self.size_A is None:
            self.size_A = 1
            for node in self.action_space:
                    self.size_A *= len(self.action_space[node])
        return self.size_S,self.size_A

    def step(self,action):
        '''
        Input an action to the environment

        Usage:
            action = {"A1":True,"A2":False}
            env.step(action)
        Input:
            action - a dict mapping action node names to values
        Output:
            state - a dict mapping state and action node names to values
            reward - the reward obtained by inputting the action
        '''
        for node in action:
            self.state[node] = action[node]
            self.state_vars[self.variables[node]] = action[node]

        # Get reward
        reward = self.factors["reward"].get_value(self.state_vars)

        return self.state,reward
