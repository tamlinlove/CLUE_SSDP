import numpy as np
from itertools import product

from Expert import Expert
from StateTable import StateTable

class PartiallyReliableExpert(Expert):
    '''
    Class for a potentially unreliable expert who may observe the whole or part
    of the state space and bases advice on what the best action would be given
    that partial observation
    '''
    def __init__(self,env,hidden_nodes,mu=10,gamma=0.01):
        '''
        Initialises a potentially unreliable expert

        Input:
            env - instance of InfluenceDiagram representing environment
            hidden_nodes - list of state node names that are not observed by the expert
            mu - interval parameter (number of trials between advice givings)
                default: 10
            gamma - tolerance parameter (regret over past trials since advice must be greater than gamma)
                default: 0.01
        '''
        self.env = env
        self.hidden_nodes = hidden_nodes
        self.mu = mu
        self.gamma = gamma

        # Fetch policy
        self.policy,self.utility = self.env.prune_state_nodes(hidden_nodes)

    def advise(self,state):
        '''
        Advise on the best action to take given an assignment of state variables

        Input:
            state - dict mapping node name to domain value

        Output:
            advice - dict mapping action node names to domain value
        '''
        index = self.policy.assignment_to_index(state)
        return self.policy.values[index]

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
        optimal_action = self.advise(state) # Get optimal action
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
        return True,optimal_action

    def get_utility(self,state,action):
        '''
        Returns the expected utility of a given state, action pair

        Input:
            state - dict mapping state node name to domain value
            action - dict mapping action node names to domain value

        Output:
            expected_utility - the expected utility of the state, action pair
        '''
        assignment = state.copy()
        for node in action:
            assignment[node] = action[node]
        index = self.utility.assignment_to_index(assignment)
        return self.utility.values[index]

    def reset(self):
        '''
        Resets the expert for a fresh run
        '''
        self.curr_trial = 0
        self.last_advice = 0
        self.optimal_utility_sum = 0
        self.agent_utility_sum = 0
