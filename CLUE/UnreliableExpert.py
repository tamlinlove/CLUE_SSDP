import numpy as np
from itertools import product

from Expert import Expert
from StateTable import StateTable

class UnreliableExpert(Expert):
    '''
    Class for a potentially unreliable expert whose probability of advising correctly
    is determined by a single parameters rho in [0,1]
    '''
    def __init__(self,env,oracle,rho,mu=10,gamma=0.01):
        '''
        Initialises a potentially unreliable expert

        Input:
            env - instance of InfluenceDiagram representing environment
            oracle - instance of TruePolicyAgent, used to get correct advice
            rho - real number between 0 and 1 inclusive, controls how often correct advice is given
                rho = 0, always incorrect
                rho = 1, always correct
            mu - interval parameter (number of trials between advice givings)
                default: 10
            gamma - tolerance parameter (regret over past trials since advice must be greater than gamma)
                default: 0.01
        '''
        self.env = env
        self.oracle = oracle
        if rho >= 0 and rho <= 1:
            self.rho = rho
        else:
            error_message = "Invalid rho = "+str(rho)+"! Must be in interval [0,1]"
            raise Exception(error_message)
        self.mu = mu
        self.gamma = gamma

        # Table for all possible actions
        self.action_table = StateTable(self.env.action_space)

    def advise(self,state,optimal_action=None):
        '''
        Advise on the best action to take given an assignment of state variables

        Input:
            state - dict mapping state node name to domain value
            optimal_action - dict mapping action node names to domain value
                if you already know the optimal action, you can pass here
                otherwise, leave as None and it will be calculated
                default: None

        Output:
            advice - dict mapping action node names to domain value
        '''
        if optimal_action is None:
            optimal_action = self.oracle.act(state) # Get best action to advise
        # Choose whether or not to give optimal advice this trial
        is_correct = np.random.choice([True,False],p=[self.rho,1-self.rho])
        if is_correct: # Give correct advice
            return optimal_action
        else: # Give incorrect advice (any advice except optimal action)
            optimal_index = self.action_table.assignment_to_index(optimal_action) # Optimal index
            indices = list(np.arange(self.action_table.size)) # Fetch all indices
            indices.remove(optimal_index) # Remove optimal index
            index = np.random.choice(indices) # Choose suboptimal index
            suboptimal_action = self.action_table.index_to_assignment(index) # Get corresponding action
            return suboptimal_action

    def deliberate(self,state,action,reward):
        '''
        Decide whether or not to give advice for this trial

        Input:
            state - dict mapping node name to domain value of pre state chance nodes
            action - dict mapping decision node names to domain value
            reward - the reward obtained this trial

        Output:
            advice_given - True if advice is given, false otherwise
            advice - dict mapping decision node names to domain value if advice_given is True, None otherwise
        '''
        # Update stats
        self.curr_trial += 1 # Increment trial counter

        # Update utility sums
        optimal_action = self.oracle.act(state) # Get optimal action
        optimal_utility = self.get_utility(state,optimal_action) # Get expected utility of optimal action
        agent_utility = self.get_utility(state,action) # Get expected utility of the agent's action
        self.optimal_utility_sum += optimal_utility # Add expected utility to running sum
        self.agent_utility_sum += agent_utility # Add expected utility to running sum

        # Condition 1: Time between trials must exceed mu
        time_difference = self.curr_trial - self.last_advice
        if time_difference < self.mu:
            return False,None

        # Condition 2: Large enough improvement is guaranteed
        utility_difference = self.optimal_utility_sum - self.agent_utility_sum
        if utility_difference/time_difference < self.gamma:
            return False,None

        # All conditions passed, giving advice
        self.last_advice = self.curr_trial # Update the trial number of the last advice giving to this trial
        self.optimal_utility_sum = 0 # Reset sums
        self.agent_utility_sum = 0 # Reset sums
        advised_action = self.advise(state,optimal_action=optimal_action) # Choose an action to advise
        return True,advised_action

    def get_utility(self,state,action):
        '''
        Returns the expected utility of a given state, action pair

        Input:
            state - dict mapping state node name to domain value
            action - dict mapping action node names to domain value

        Output:
            expected_utility - the expected utility of the state, action pair
        '''
        return self.oracle.expected_utility(state,action)


    def reset(self):
        '''
        Resets the expert for a fresh run
        '''
        self.curr_trial = 0
        self.last_advice = 0
        self.optimal_utility_sum = 0
        self.agent_utility_sum = 0
