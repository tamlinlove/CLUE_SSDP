from Agent import Agent
from BaselineAgent import BaselineAgent
from StateTable import StateTable

import numpy as np

class NaiveAdviceFollower(Agent):
    '''
    Class for NAF agent that always follows advice it receives
    If it gets more than one piece of advice for a state, will randomly choose
    Otherwise, acts like the agent it wraps around
    '''
    def __init__(self,env,agent=None,trials=None,**kwargs):
        '''
        Initialise NAF agent

        Input:
            env - instance of InfluenceDiagram class representing environment
            agent - instance of the Agent class or None
                if Agent, NAF wraps around that agent and uses it for decision making and learning
                if None, will create BaselineAgent with default parameters
                default: None
            trials - number of trials, only required if not specifying agent
                default: None
        '''
        self.name = "NAF"
        self.state_space = env.state_space
        if agent is None: # Default agent
            if isinstance(trials,int) and trials > 0: # Trials is valid
                self.agent = BaselineAgent(env,trials)
            else: # Invalid trials
                error_message = str(trials)+" is an invalid number of trials (must be int > 0)"
                raise Exception(error_message)
        else: # Agent has been specified
            if isinstance(agent,Agent): # agent is a valid agent
                self.agent = agent
            else:
                error_message = "Invalid agent passed to NAF"
                raise Exception(error_message)

    def act(self,state,explore=True):
        '''
        Select an action given a state

        Input:
            state - dict mapping state variable names to values
            explore - whether or not the agent should explore
                default: True
        Output:
            action - dict mapping action variable names to values
        '''
        # Check if we have been advised previously
        advice_list = self.aggregate_advice(state)
        if len(advice_list) == 0: # No advice received thusfar, act normally
            return self.agent.act(state,explore)
        else: # Advice received for this state
            return np.random.choice(advice_list) # Randomly choose from given advice

    def aggregate_advice(self,state):
        '''
        Retrieve all advice for a given state and put it in a list

        Input:
            state - dict mapping state node name to value
        Output:
            advice_list - list of advice, each advice is an action dict
        '''
        advice_list = []
        for expert in self.state_table_dict:
            advice = self.state_table_dict[expert].get_value(state)
            if advice is not None:
                advice_list.append(advice)
        return advice_list

    '''
    Do some learning given an assigment of state and action and the reward

    Input:
        state - a dict mapping state node name to value in the node's domain
        actions - a dict mapping action node names to values in the node's domain
        reward - the reward of this trial
        advice - a dict mapping expert name to advice, where advice is an action dict

    '''
    def learn(self,state,actions,reward,advice):
        # Learn
        self.agent.learn(state,actions,reward)
        # Add advice to state table
        for expert in self.state_table_dict:
            if advice[expert] is not None:
                self.state_table_dict[expert].add_to_table(state,advice[expert])

    def reset(self,panel):
        '''
        Reset the agent for a fresh run

        Input:
            panel - a panel of experts, instance of Panel class
        '''
        self.state_table_dict = {}
        for expert in panel.experts:
            self.state_table_dict[expert] = StateTable(self.state_space)
        self.agent.reset()

    def takes_advice(self):
        '''
        Whether or not an agent takes advice

        Output:
            True if the agent incorporates expert advice when learning/acting
            False otherwise
        '''
        return True
