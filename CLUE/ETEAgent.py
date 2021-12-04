import numpy as np
from itertools import product

from Agent import Agent
from Utility import Utility

class ETEAgent(Agent):
    '''
    Class for an action-value epsilon-greedy SSDP agent
    '''
    def __init__(self,env,trials,exp_trials=None,Q0=0,alpha=None,**kwargs):
        '''
        Initialise action-value epsilon-greedy SSDP agent

        Input:
            env - instance of InfluenceDiagram class representing the environment
            trials - the number of trials over which the agent will learn
            eps_start - the starting value of epsilon
                default: 1
            eps_end - the end value of epsilon
                default: 0
            eps_fraction - the fractions of trials over which epsilon decays from start to end
                default: 0.8
            Q0 - initial Q value. Can be real number or array of appropriate size
                default: 0
            alpha - learning rate. Real number or None
                if None, will be based on visit count (see Sutton and Barto, 2018)
                default: None
        '''
        self.name = "ETE Baseline Agent"
        # Initialise state and action space
        self.state_space = env.state_space
        self.action_space = env.action_space

        # Initialise dictionary of node types
        self.node_types = env.node_types
        self.domains = env.domains

        # Parameters
        self.trials = trials
        self.Q0 = Q0
        self.alpha = alpha
        if exp_trials is None:
            self.exp_trials = trials/4
        else:
            self.exp_trials = exp_trials

    def act(self,state,explore=True,return_exploit=False):
        '''
        Select an action given a state

        Input:
            state - dict mapping state variable names to values
            explore - whether or not the agent should explore
                default: True
            return_exploit = whether or not the function will return the value of exploit
                default: False
        Output:
            action - dict mapping action variable names to values
            exploit - boolean. If True, the agent exploited, else explored.
                Only returned if return_exploit is True
        '''
        if self.trial_count < self.exp_trials:
            exploit = False
        else:
            exploit = True

        # Select action
        action = {}
        if exploit:
            # Select best action according to utility estimate
            _,best_action = self.exploit(state)
            for node in self.action_space:
                action[node] = best_action[node]
        else:
            # Randomly make decisions
            for node in self.action_space:
                action[node] = np.random.choice(self.action_space[node])
        if return_exploit:
            return action,exploit
        return action

    def exploit(self,state):
        '''
        Select the best action

        Input:
            state - dict mapping state variable name to value
        Output:
            best_index - index of best state-action pair in utility
            best_action - dict mapping node name to domain value of state-action pair
        '''
        state_values,indices = self.score_state(state) # Score each action for the state
        best_index = indices[np.argmax(state_values)] # Select the best scoring action index
        best_action = self.utility.index_to_assignment(best_index) # Find corresponding action
        return best_index,best_action

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

    def reset(self):
        '''
        Reset agent for a fresh run
        '''
        self.trial_count = 0 # Reset number of trials
        self.utility = Utility(self.domains,self.Q0,self.alpha) # Reset utility

    def score_state(self,state):
        '''
        Calculates the expected utility for each action for a given state

        Input:
            state - dict mapping state variable names to values
        Output:
            state_values - list of values
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
        state_values = []
        for index in indices:
            state_values.append(self.utility.values[index])
        return state_values,indices
