from Agent import Agent
from BaselineAgent import BaselineAgent
from StateTable import StateTable

import numpy as np
from itertools import product

class PRQAgent(Agent):

    def __init__(self,env,agent=None,trials=None,temperature=0,temperature_change=0.05,**kwargs):
        self.name = "PRQ"
        self.state_space = env.state_space
        self.action_space = env.action_space
        self.initial_temp = temperature
        self.d_temp = temperature_change

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
                error_message = "Invalid agent passed to PRQ"
                raise Exception(error_message)

    def act(self,state,explore=True):
        # Check if agent has been advised previously
        advice_dict = self.aggregate_advice(state)
        advice_given = bool(advice_dict)

        if advice_given:
            # Calculate probability of using P
            e_pows = [np.e**(self.temp**self.w_task)]
            for expert in advice_dict:
                e_pows.append(np.e**(self.temp**self.w_experts[expert]))
            probs = e_pows/np.sum(e_pows)

            policies = ["myself"]+list(advice_dict.keys())

            self.chosen_policy = np.random.choice(policies,p=probs)

            if self.chosen_policy == "myself":
                # Agent's policy
                return self.agent.act(state)
            else:
                return advice_dict[self.chosen_policy]
        else:
            self.chosen_policy = "myself"
            return self.agent.act(state)

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

    def learn(self,state,actions,reward,advice):
        # Learn
        self.agent.learn(state,actions,reward)
        # Add advice to state table
        for expert in self.state_table_dict:
            # Add advice to table
            if advice[expert] is not None:
                self.state_table_dict[expert].add_to_table(state,advice[expert])
        # PRQ Stuff
        if self.chosen_policy == "myself":
            self.w_task = (self.w_task*self.u_task+reward)/(self.u_task+1)
            self.u_task += 1
        else:
            self.w_experts[self.chosen_policy] = (self.w_experts[self.chosen_policy]*self.u_experts[self.chosen_policy]+reward)/(self.u_experts[self.chosen_policy]+1)
            self.u_experts[self.chosen_policy] += 1
        self.temp += self.d_temp


    def reset(self,panel):
        self.agent.reset() # Reset agent
        # PRQ Stuff
        self.state_table_dict = {} # Reset advice table
        self.temp = self.initial_temp
        self.w_task = 0
        self.u_task = 0
        self.w_experts = {}
        self.u_experts = {}
        #self.chosen_policy = None
        for expert in panel.experts:
            self.state_table_dict[expert] = StateTable(self.state_space)
            self.w_experts[expert] = 0
            self.u_experts[expert] = 0

    def takes_advice(self):
        '''
        Whether or not an agent takes advice

        Output:
            True if the agent incorporates expert advice when learning/acting
            False otherwise
        '''
        return True
