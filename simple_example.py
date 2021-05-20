import CLUE
import numpy as np

'''
ENVIRONMENT
'''
# Create a random environment with 7 state variables (|S|=128)
# and 3 action variables (|A|=8)
env = CLUE.make("RandomSSDP",num_chance=7,num_decision=3)

'''
EXPERIMENT DETAILS
'''
trials = 10000 # Number of trials each run
runs = 5 # Number of runs, each run the agent learns from scratch

'''
AGENTS
'''
# Create a True Policy Agent
true_policy_agent = CLUE.TruePolicyAgent(env)

# Create a Baseline (action-value epsilon-greedy) Agent
baseline_agent = CLUE.BaselineAgent(env,trials)

# Create a Naive Advice Follower (NAF) with a default action-value epsilon-greedy agent
naf_agent = CLUE.NaiveAdviceFollower(env,trials=trials)

# Cretae a CLUE Agent with a default action-value epsilon-greedy agent
clue_agent = CLUE.ClueAgent(env,trials=trials)

# Add all to a list
agents = [true_policy_agent,baseline_agent,naf_agent,clue_agent]

'''
PANEL OF EXPERTS
'''
#TODO

'''
RUN EXPERIMENT
'''
#TODO
