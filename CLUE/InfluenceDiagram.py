'''
Parent class for any SSDP influence diagram environment
'''

import numpy as np
import PGM
import StateTable

class InfluenceDiagram():
    '''
    Parent class for any influence diagram environment

    Assumes:
        No decision variable is a parent to a chance variable
        A single utility node, called "reward"

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

    def prune_state_nodes(self,state_nodes):
        '''
        Removes state nodes from the ID and adjusts CPDs accordingly

        Input:
            state_nodes - list of chance node names
        Output:
            policy - StateTable object whose domain is the state space (without state_nodes)
                and each state assignment maps to an action assignment
            utility - StateTable object mapping state-action assignments to expected utilities
        '''
        # Get list of all factors
        fac_list = list(self.factors.values())

        # Eliminate all nodes in stet_nodes
        ve_engine = PGM.VE(self.gm)
        new_facs = fac_list
        new_state_space = self.state_space.copy()
        for node in state_nodes:
            new_facs = ve_engine.eliminate_var(new_facs,self.variables[node])
            new_state_space.pop(node,None)
        reward_fac = new_facs[-1] # Final factor is the factor associated with reward

        # State-Action Space
        state_action_space = new_state_space.copy()
        for node in self.action_space:
            state_action_space[node] = self.action_space[node]

        # Create policy object
        policy = StateTable.StateTable(new_state_space)
        utility = StateTable.StateTable(state_action_space)
        action_assignments = StateTable.StateTable(self.action_space)

        # For every state, which action maximises reward?
        for state_index in range(policy.size):
            assignment = policy.index_to_assignment(state_index)
            node_assignment = assignment.copy()
            var_assignment = {}
            for node in assignment:
                var_assignment[self.variables[node]] = assignment[node]
            best_action = None
            best_reward = None
            for action_index in range(action_assignments.size):
                action_assignment = action_assignments.index_to_assignment(action_index)
                for action_node in self.decision:
                    node_assignment[action_node] = action_assignment[action_node]
                    var_assignment[self.variables[action_node]] = action_assignment[action_node]

                reward = reward_fac.get_value(var_assignment)
                utility.values[utility.assignment_to_index(node_assignment)] = reward
                if best_reward is None or best_reward < reward:
                    best_action = action_assignment
                    best_reward = reward
            policy.add_to_table(assignment,best_action)
        return policy,utility

    def reset(self):
        '''
        Resets the environments current state and samples a new trial
        '''
        self.state = {} # Reset to empty state
        self.state_vars = {} # Reset variables
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

    def step(self,action,state=None):
        '''
        Input an action to the environment

        Usage:
            action = {"A1":True,"A2":False}
            reward = env.step(action)
        Input:
            action - a dict mapping action node names to values
            state - a dict mapping state node names to values
                if None, will use the existing state set by the previous call to sample
                if a dict, will replace the existing state
                default: None
        Output:
            reward - the reward obtained by inputting the action
        '''
        if state is not None:
            self.state = state
            for node in state:
                self.state_vars[self.variables[node]] = self.state[node]

        for node in action:
            self.state_vars[self.variables[node]] = action[node]

        # Get reward
        reward = self.factors["reward"].get_value(self.state_vars)

        return reward
