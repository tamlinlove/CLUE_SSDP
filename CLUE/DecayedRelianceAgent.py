from Agent import Agent
from BaselineAgent import BaselineAgent
from StateTable import StateTable

import numpy as np

class DecayedRelianceAgent(Agent):
    '''
    Class for agent that initially follows all advice but relies on it less as time goes on
    '''
    def __init__(self,env,trials,agent=None,initial_reliance=1,final_reliance=0,reliance_fraction=0.8,**kwargs):
        '''
        Initialise NAF agent

        Input:
            env - instance of InfluenceDiagram class representing environment
            trials - number of trials
            agent - instance of the Agent class or None
                if Agent, DRA wraps around that agent and uses it for decision making and learning
                if None, will create BaselineAgent with default parameters
                default: None
            initial_reliance - initial probability that agent follows advice, between 0 and 1 inclusive
                default: 1
            final_reliance - final probability that agent follows advice between, 0 and 1 inclusive
                default: 0
            reliance_fraction - percentage of trials over which reliance decays, between 0 and 1 inclusive
                default: 0.8

        '''
        self.name = "DecayedRelianceAgent"
        self.state_space = env.state_space

        # Validation for trials
        if isinstance(trials,int) and trials > 0: # Trials is valid
            self.trials = trials
        else: # Invalid trials
            error_message = str(trials)+" is an invalid number of trials (must be int > 0)"
            raise Exception(error_message)

        if agent is None: # Default agent
            self.agent = BaselineAgent(env,trials)
        else: # Agent has been specified
            if isinstance(agent,Agent): # agent is a valid agent
                self.agent = agent
            else:
                error_message = "Invalid agent passed to DecayedRelianceAgent"
                raise Exception(error_message)

        # Validation for reliance
        if initial_reliance >= 0 and initial_reliance <= 1:
            self.initial_reliance = initial_reliance
        else:
            error_message = "initial_reliance " + str(initial_reliance) + " must be on interval [0,1]"
            raise Exception(error_message)

        if final_reliance >= 0 and final_reliance <= 1:
            if final_reliance <= initial_reliance:
                self.final_reliance = final_reliance
            else:
                error_message = "final_reliance " + str(final_reliance) + " must be smaller than initial_reliance "+str(initial_reliance)
                raise Exception(error_message)
        else:
            error_message = "final_reliance " + str(final_reliance) + " must be on interval [0,1]"
            raise Exception(error_message)

        if reliance_fraction >= 0 and reliance_fraction <= 1:
            self.reliance_fraction = reliance_fraction
        else:
            error_message = "reliance_fraction " + str(reliance_fraction) + " must be on interval [0,1]"
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

        # Recalculate reliance
        fraction = min(1.0, self.trial_count / self.reliance_steps)
        self.reliance = self.initial_reliance + fraction * (self.final_reliance - self.initial_reliance)

        if len(advice_list) == 0: # No advice received thusfar, act normally
            return self.agent.act(state,explore)
        else: # Advice received for this state
            follows_advice = np.random.choice([True,False],p=[self.reliance,1-self.reliance])
            if follows_advice: # Follow advice
                return np.random.choice(advice_list) # Randomly choose from given advice
            else: # Act unassisted
                return self.agent.act(state,explore)

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
        # Update trial count
        self.trial_count += 1
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
        self.trial_count = 0 # Reset number of trials
        self.reliance = self.initial_reliance # Reset reliance on advice
        self.reliance_steps = self.reliance_fraction * float(self.trials) # Reset reliance decay


    def takes_advice(self):
        '''
        Whether or not an agent takes advice

        Output:
            True if the agent incorporates expert advice when learning/acting
            False otherwise
        '''
        return True
