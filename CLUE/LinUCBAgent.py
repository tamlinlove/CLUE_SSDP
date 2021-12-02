import numpy as np
from itertools import product

from Agent import Agent
from Utility import Utility
from StateTable import StateTable

class LinUCBAgent(Agent):
    '''
    Class for LinUCB agent
    Assumes the domain of each state variable to be the boolean True or False
    '''
    def __init__(self,env,alpha=2,**kwargs):
        '''
        Initialise LinUCB agent

        Input:
            env - instance of InfluenceDiagram class representing the environment
            alpha - TODO
        '''
        self.name = "LinUCB Baseline Agent"
        # Initialise state and action space
        self.state_space = env.state_space
        self.action_space = env.action_space

        # Initialise dictionary of node types
        self.node_types = env.node_types
        self.domains = env.domains
        self.chance = env.chance
        self.decision = env.decision
        self.dimensions = len(env.chance)

        # Set up offsets and size
        size = 1
        self.offsets = {}
        for i in range(len(self.decision)-1,-1,-1):
            self.offsets[self.decision[i]]=size
            size *= len(self.action_space[self.decision[i]]) # Offset by size of domain
        self.arms = size

        # Parameters
        self.alpha = alpha

        self.reset()

    def act(self,state,**kwargs):
        '''
        Select an action given a state

        Input:
            state - dict mapping state variable names to values
        Output:
            action - dict mapping action variable names to values
        '''
        max_ucb = -1
        ucbs = self.UCB(state)
        max_arms = np.argwhere(ucbs == np.amax(ucbs)).flatten().tolist()
        action_index = np.random.choice(max_arms)
        return self.index_to_action(action_index)

    def action_to_index(self,action):
        '''
        Get the index of the appropriate arm for a given action

        Input:
            action - a dict mapping action node names to values
        Output:
            index - index of the appropriate arm
        '''
        index = 0
        for node in self.action_space:
            index += self.action_space[node].index(action[node])*self.offsets[node]
        return index

    def index_to_action(self,index):
        '''
        Get the action node assignment for a given arm index

        Input:
            index - index of the appropriate arm
        Output:
            action - a dict mapping action node names to values
        '''
        action = {}
        for i in range(len(self.decision)-1,-1,-1):
            action[self.decision[i]] = self.action_space[self.decision[i]][index % len(self.action_space[self.decision[i]])]
            index = index // len(self.action_space[self.decision[i]])
        return action


    def learn(self,state,action,reward):
        '''
        Update learned parameters for chosen arm

        Input:
            state - a dict mapping state node names to values
            action - a dict mapping action node names to values
            reward - the reward of this trial
        '''
        reward = (reward + 1)/2 # Move from [-1,1] to [0,1]
        x = self.state_to_vector(state)
        a = self.action_to_index(action)
        self.A[a] += np.dot(x, x.T)
        self.b[a] += reward * x



    def reset(self):
        '''
        Reset agent for a fresh run
        '''
        self.A = []
        self.b = []
        for i in range(self.arms):
            self.A.append(np.identity(self.dimensions))
            self.b.append(np.zeros([self.dimensions,1]))

    def state_to_vector(self,state):
        '''
        Convert the state representation into a normalised vector of |Vs|x1
        Assumes the domain of each state variable to be a boolean

        Input:
            state - dict mapping state variable names to values
        Output:
            x - a normalised |Vs|x1 vector representation of the state
        '''
        x = np.zeros([self.dimensions,1])
        for i in range(self.dimensions):
            if state[self.chance[i]]:
                x[i] = 1
            else:
                x[i] = 0
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        else:
            return x/norm

    def vector_to_state(self,vector):
        '''
        TODO
        '''
        state = {}
        for i in range(self.dimensions):
            if vector[i] == 0:
                state[self.chance[i]] = False
            else:
                state[self.chance[i]] = True
        return state


    def UCB(self,state):
        '''
        Calculate the UCB for each arm given the state

        Input:
            state - dict mapping state variable names to values
        Output:
            p - a list of UCB values
        '''
        p = []
        x = self.state_to_vector(state)
        for a in range(self.arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = np.dot(A_inv, self.b[a])
            ucb = np.dot(theta.T,x) +  self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv,x)))
            ucb = ucb[0][0]
            p.append(ucb)
        return p
