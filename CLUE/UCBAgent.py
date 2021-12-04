import numpy as np
from itertools import product

from Agent import Agent
from Utility import Utility
from StateTable import StateTable

class UCBAgent(Agent):
    '''
    Class for a UCBAgent
    '''
    def __init__(self,env,Q0=0,alpha=None,c=1,**kwargs):
        '''
        Initialise action-value epsilon-greedy SSDP agent

        Input:
            env - instance of InfluenceDiagram class representing the environment
            Q0 - initial Q value. Can be real number or array of appropriate size
                default: 0
            alpha - learning rate. Real number or None
                if None, will be based on visit count (see Sutton and Barto, 2018)
                default: None
        '''
        self.name = "UCB_Baseline_Agent"
        # Initialise state and action space
        self.state_space = env.state_space
        self.action_space = env.action_space

        # Initialise dictionary of node types
        self.node_types = env.node_types
        self.domains = env.domains

        # Parameters
        self.Q0 = Q0
        self.alpha = alpha
        self.c = c

    def act(self,state,return_exploit=False):
        '''
        Select an action given a state

        Input:
            state - dict mapping state variable names to values
            return_exploit = whether or not the function will return the value of exploit
                default: False
        Output:
            action - dict mapping action variable names to values
            exploit - boolean. If True, the agent exploited, else explored.
                Only returned if return_exploit is True
        '''
        ucb_vals,utility_vals,indices = self.score_state(state,return_ucb=True) # Score each action for the state
        best_index = indices[np.argmax(ucb_vals)] # Select the best scoring action index
        best_action = self.utility.index_to_assignment(best_index) # Find corresponding action
        action = {}
        for node in self.action_space:
            action[node] = best_action[node]
        if return_exploit:
            optimal_index = indices[np.argmax(utility_vals)]
            if optimal_index == best_index:
                # UCB chose the best action
                return action,True
            else:
                # UCB chose a suboptimal action
                return action,False
        return action

    def learn(self,state,actions,reward):
        '''
        Update action-value function based on state-action-reward

        Input:
            state - a dict mapping state node names to values
            actions - a dict mapping action node names to values
            reward - the reward of this trial
        '''
        # Update trial count
        self.trial_count += 1
        # Pack state, action and reward into a trial
        trial = {}
        for node in state:
            trial[node] = state[node]
        for node in actions:
            trial[node] = actions[node]
        trial["reward"] = reward
        # Update utility with trial
        self.utility.update(trial)
        # Update visit counts
        self.visits.add_to_table(trial,self.visits.get_value(trial)+1)

    def reset(self):
        '''
        Reset agent for a fresh run
        '''
        self.trial_count = 0
        self.utility = Utility(self.domains,self.Q0,self.alpha) # Reset utility
        self.visits = StateTable(self.domains,default_value=1) # Reset visit counts

    def score_state(self,state,return_ucb=False):
        '''
        Calculates the UCB score for each action for a given state

        Input:
            state - dict mapping state variable names to values
            return_ucb - bool. If True, will return UCB score as well as utility score
                default: False
        Output:
            ucb_vals - list of UCB values (i.e. Q(s,a) + c sqrt(log(t)/N(s,a)) for each a)
                Only returned if return_ucb is True
            utility_vals - list of utility values (i.e. Q(s,a) for each a)
            indices - list of indices corresponding to each value
        '''
        combinations = list(product(*self.action_space.values())) # All possible actions
        keys = list(self.action_space.keys())
        indices = []
        # Get all indices
        for combo in combinations:
            state_copy = state.copy()
            for i in range(len(keys)):
                state_copy[keys[i]] = combo[i]
            indices.append(self.utility.assignment_to_index(state_copy))
        # Get all values
        ucb_vals = []
        utility_vals = []
        for index in indices:
            exploit_term = self.utility.values[index]
            utility_vals.append(exploit_term)
            explore_term = self.c*(np.sqrt(np.log(self.trial_count+1)/(self.visits.values[index])))
            score = exploit_term + explore_term
            ucb_vals.append(score)
        if return_ucb:
            return ucb_vals,utility_vals,indices
        return utility_vals,indices
