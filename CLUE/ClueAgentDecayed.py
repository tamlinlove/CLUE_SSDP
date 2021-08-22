from Agent import Agent
from BaselineAgent import BaselineAgent
from StateTable import StateTable

import numpy as np
from itertools import product

class ClueAgentDecayed(Agent):
    '''
    Class for a CLUE (Cautiously Learning with Unreliable Experts) agent
    Takes advice and combines advice in Bayesian way
    Reliability for all starts at 1 and decays to 0
    '''
    def __init__(self,env,trials,agent=None,initial_reliance=1,reliance_fraction=0.8,threshold=None,**kwargs):
        '''
        Initialise CLUE Agent

        Input:
            env - instance of InfluenceDiagram class representing environment
            trials - number of trials, required for decaying to work
            agent - instance of the Agent class or None
                if Agent, CLUE wraps around that agent and uses it for decision making and learning
                if None, will create BaselineAgent with default parameters
                default: None

            initial_reliance - initial reliance on advice between 0 and 1 (none and all)
            reliance_fraction - fraction of trials over which the reliance decays
            threshold - threshold parameter T, used in decision making
                default: min(2/|A|,0.5)
        '''
        self.name = "CLUE with Decayed Reliance"
        self.state_space = env.state_space
        self.action_space = env.action_space
        self.initial_reliance = initial_reliance
        self.reliance_fraction = reliance_fraction
        self.trials = trials

        if threshold is None:
            self.threshold = min(2/env.size_A,0.5)
        else:
            self.threshold = threshold

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
                error_message = "Invalid agent passed to CLUE"
                raise Exception(error_message)
        self.history = {"rho":{}}

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
        # Check if agent has been advised previously
        advice_dict = self.aggregate_advice(state)
        advice_given = bool(advice_dict)
        # Calculate trust in each expert
        if advice_given:
            # Recalculate rho
            fraction = min(1.0, self.trial_count / self.reliance_steps)
            self.reliance = self.initial_reliance + fraction * (0 - self.initial_reliance)

            for expert in self.state_table_dict:
                self.rho[expert] = self.reliance
                self.history["rho"][expert].append(self.reliance) # Add new rho to rho history

            # Calculate best advised action
            combinations = list(product(*self.action_space.values()))
            keys = list(self.action_space.keys())
            # Calculate likelihood terms
            likelihoods = np.zeros(len(combinations))
            for i in range(len(combinations)):
                likelihood = 1
                for expert in advice_dict:
                    all_same = True
                    # Check if expert has advised this action
                    for j in range(len(keys)):
                        if advice_dict[expert][keys[j]] != combinations[i][j]:
                            all_same = False
                            break
                    if all_same:
                        likelihood *= self.rho[expert] # Expert advised action
                    else:
                        likelihood *= (1-self.rho[expert])/(len(combinations)-1) # Expert did not advise action
                likelihoods[i] = likelihood
            if np.sum(likelihoods)==0:
                # Something is wrong, ignore for now
                trust = False
            else:
                # Calculate probability for each action
                probs = np.zeros(len(combinations))
                for i in range(len(combinations)):
                    probs[i] = likelihoods[i]/np.sum(likelihoods)
                best_index = np.argmax(probs)
                # Choose whether to follow advice or act epsilon greedy
                if probs[best_index]<self.threshold:
                    trust = False
                else:
                    trust = np.random.choice([True,False],p=[probs[best_index],1-probs[best_index]])
        else:
            trust = False
            for expert in self.state_table_dict:
                self.history["rho"][expert].append(self.rho[expert]) # No advice, so add last known rho to history

        # Act
        if explore and trust: # Follow advice
            action = {}
            for i in range(len(keys)):
                action[keys[i]] = combinations[best_index][i]
            return action
        else: # Don't follow advice
            return self.agent.act(state,explore)

    def aggregate_advice(self,state):
        '''
        Retrieve all advice for a given state and put it in a dictionary

        Input:
            state - dict mapping state node name to value
        Output:
            advice_dict - dict mapping expert name to advice
        '''
        advice_dict = {}
        for expert in self.state_table_dict:
            advice = self.state_table_dict[expert].get_value(state)
            if advice is not None:
                advice_dict[expert] = advice
        return advice_dict

    def get_history(self):
        '''
        Returns history dict. This contains history of rho values

        Output:
            history - a dict of history
        '''
        return self.history

    def learn(self,state,actions,reward,advice):
        '''
        Do some learning given an assigment of variables and the reward

        Input:
            state - a dict mapping state node name to value in the node's domain
            actions - a dict mapping action node names to values in the node's domain
            reward - the reward of this trial
            advice - a dict mapping expert name to advice, where advice is an action dict

        '''
        self.trial_count += 1
        # Learn
        self.agent.learn(state,actions,reward)
        # Add advice to state table
        for expert in self.state_table_dict:
            if advice[expert] is not None: # Advice was given this trial
                # Add advice to table
                self.state_table_dict[expert].add_to_table(state,advice[expert])


    def reset(self,panel):
        '''
        Reset the agent for a fresh run

        Input:
            panel - a panel of experts, instance of Panel class
        '''
        self.agent.reset() # Reset agent
        self.history = {"rho":{}} # Reset history
        self.state_table_dict = {} # Reset advice table
        self.rho = {}
        self.trial_count = 0
        self.reliance = self.initial_reliance # Reset epsilon
        self.reliance_steps = self.reliance_fraction * float(self.trials) # Reset epsilon decay
        for expert in panel.experts:
            self.state_table_dict[expert] = StateTable(self.state_space)
            self.history["rho"][expert] = []
            self.rho[expert] = self.initial_reliance

    def takes_advice(self):
        '''
        Whether or not an agent takes advice

        Output:
            True if the agent incorporates expert advice when learning/acting
            False otherwise
        '''
        return True
