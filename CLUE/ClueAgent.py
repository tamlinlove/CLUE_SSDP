from Agent import Agent
from BaselineAgent import BaselineAgent
from StateTable import StateTable

import numpy as np
from itertools import product


class ClueAgent(Agent):
    '''
    Class for a CLUE (Cautiously Learning with Unreliable Experts) agent
    Takes advice, estimates reliability of experts and combines advice in Bayesian way
    '''
    def __init__(
        self,
        env,
        trials,
        agent=None,
        initial_estimate=[1,1],
        eps_start=1,
        eps_end=0,
        eps_fraction=0.8,
        threshold=None,
        no_bayes=False,
        regular_update=True,
        sliding_window=None,
        recency=None,
        **kwargs):
        '''
        Initialise CLUE Agent

        Input:
            env - instance of InfluenceDiagram class representing environment
            agent - instance of the Agent class or None
                if Agent, CLUE wraps around that agent and uses it for decision making and learning
                if None, will create BaselineAgent with default parameters
                default: None
            trials - number of trials, only required if not specifying agent
                default: None
            initial_estimate - list of two parameters, alpha and beta, for initial beta distribution for each expert
                default: [1,1] (uniform prior)
                TODO: allow for each expert to have a separate initial_beta
            threshold - threshold parameter T, used in decision making
                default: 2/|A|
            no_bayes - if True, will only follow advice of most reliable expert, discarding the rest
                    if False, will act as a regular CLUE agent, combining advice through Bayes rule
                default: False
            regular_update - if False, will only evaluate new advice
                if True, will evaluate advice it received for a state every time it visits that state
                default: True
            sliding_window - the number of previous evaluations used to estimate reliability
                if None, no sliding window will be used and all evaluations count equally
                default: None
            recency - If None, beta distribution update will weight all observations equally
                    If not None, beta distribution update will weight recent observations more
                    rho = (1-recency)*rho + recency*(alpha/(alpha+beta))
        '''
        self.name = "CLUE"
        self.state_space = env.state_space
        self.action_space = env.action_space
        self.initial_estimate = initial_estimate
        self.no_bayes = no_bayes
        self.regular_update = regular_update
        self.recency = recency

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_fraction = eps_fraction

        self.trials = trials

        if no_bayes:
            self.name += " (Naive)"
        if threshold is None:
            self.threshold = min(2/env.size_A,0.5) # Ensures that auto threshold is always below 1 for when |A| is small
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


        if sliding_window is None or isinstance(sliding_window,int):
            self.sliding_window = sliding_window
        else:
            error_message = "sliding_window must be None or int, not "+str(sliding_window)
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
        # Recalculate epsilon
        fraction = min(1.0, self.trial_count / self.epsilon_steps)
        self.epsilon = self.eps_start + fraction * (self.eps_end - self.eps_start)

        # Check if agent has been advised previously
        advice_dict = self.aggregate_advice(state)
        advice_given = bool(advice_dict)
        best_action = None

        # Choose whether to exploit or explore
        if explore:
            exploit = np.random.choice([True,False],p=[1-self.epsilon,self.epsilon]) # Maybe explore
        else:
            exploit = True # Always exploit

        for expert in self.state_table_dict:
            if self.recency is not None:
                rho = max(self.recent_rho[expert],0) # Just in case
                self.rho[expert] = rho
                self.history["rho"][expert].append(rho) # Add new rho to rho history
            else:
                alpha = self.beta_parameters[expert][0]
                beta = self.beta_parameters[expert][1]
                rho = alpha/(alpha+beta)
                rho = max(rho,0) # Just in case
                #rho = rho**2
                self.rho[expert] = rho
                self.history["rho"][expert].append(rho) # Add new rho to rho history

        if exploit:
            return self.agent.act(state,explore=False)
        else:
            # Calculate trust in each expert
            if advice_given:
                if self.no_bayes:
                    # Select the action advised by the best expert, with probability E[rho(expert)]
                    # Ignore consensus, etc.
                    best_expert = max(self.rho, key=self.rho.get)
                    trust = np.random.choice([True,False],p=[self.rho[best_expert],1-self.rho[best_expert]])
                    if trust:
                        best_action = advice_dict[best_expert]
                else:
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
                            if trust:
                                best_action = {}
                                for i in range(len(keys)):
                                    best_action[keys[i]] = combinations[best_index][i]
            else:
                trust = False
            # Act
            if trust and best_action is not None: # Follow advice
                return best_action
            else: # Don't follow advice
                action = {}
                # Randomly make decisions
                for node in self.action_space:
                    action[node] = np.random.choice(self.action_space[node])
                return action

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
            # Add advice to table
            if advice[expert] is not None:
                self.state_table_dict[expert].add_to_table(state,advice[expert])

            # Decide between regular update or update only on receiving new advice
            if self.regular_update:
                # Fetch the latest advice (could be this trial or earlier)
                latest_advice = self.state_table_dict[expert].get_value(state)
            else:
                # Only fetch advice from this trial (could be None)
                latest_advice = advice[expert]

            if latest_advice is not None: # Can update this trial
                # Evaluate advice
                state_values,indices = self.agent.score_state(state)
                best_val = max(state_values)
                worst_val = min(state_values)
                advice_state = {}
                for node in self.state_space:
                    advice_state[node] = state[node]
                for node in actions:
                    advice_state[node] = latest_advice[node]
                advice_val = self.agent.utility.get_value(advice_state)
                if self.sliding_window is not None: # Only count sliding_window number of evaluations
                    if advice_val >= best_val:
                        self.evals[expert].append(1) # Expert's advice is best
                    else:
                        self.evals[expert].append(0) # Expert's advice is not optimal

                    if len(self.evals[expert])>self.sliding_window:
                        self.evals[expert] = self.evals[expert][-self.sliding_window:]

                    self.optimal_dict[expert] = np.sum(self.evals[expert])
                    self.suboptimal_dict[expert] = len(self.evals[expert])-self.optimal_dict[expert]

                else: # Count all evaluations
                    if advice_val >= best_val:
                        self.optimal_dict[expert] += 1 # Expert's advice is best
                    else:
                        self.suboptimal_dict[expert] += 1 # Expert's advice is not optimal
                # Update parameters
                self.beta_parameters[expert] = (self.optimal_dict[expert],self.suboptimal_dict[expert])
                if self.recency is not None:
                    # Recency weighted moving average
                    if advice_val >= best_val:
                        count_update = 1
                    else:
                        count_update = 0
                    self.recent_rho[expert] = (1-self.recency) * self.recent_rho[expert] + count_update * self.recency


    def reset(self,panel):
        '''
        Reset the agent for a fresh run

        Input:
            panel - a panel of experts, instance of Panel class
        '''
        self.agent.reset() # Reset agent
        self.trial_count = 0 # Reset number of trials
        self.epsilon = self.eps_start # Reset epsilon
        self.epsilon_steps = self.eps_fraction * float(self.trials) # Reset epsilon decay
        self.history = {"rho":{}} # Reset history
        self.state_table_dict = {} # Reset advice table
        self.beta_parameters = {} # Reset reliability estimates
        self.optimal_dict = {}
        self.suboptimal_dict = {}
        self.rho = {}
        self.recent_rho = {}
        self.evals = {}
        for expert in panel.experts:
            self.state_table_dict[expert] = StateTable(self.state_space)
            self.beta_parameters[expert] = (self.initial_estimate[0],self.initial_estimate[1])
            self.optimal_dict[expert] = self.initial_estimate[0]
            self.suboptimal_dict[expert] = self.initial_estimate[1]
            self.history["rho"][expert] = []
            self.rho[expert] = self.initial_estimate[0]/(self.initial_estimate[0]+self.initial_estimate[1])
            self.recent_rho[expert] = self.rho[expert]
            if self.sliding_window is not None:
                self.evals[expert] = []

    def takes_advice(self):
        '''
        Whether or not an agent takes advice

        Output:
            True if the agent incorporates expert advice when learning/acting
            False otherwise
        '''
        return True
