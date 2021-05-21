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
# Create some agents, store in dictionary
agents = {
    "True Policy Agent":CLUE.TruePolicyAgent(env), # True policy agent
    "Baseline Agent":CLUE.BaselineAgent(env,trials), # Action-value epsilon-greedy
    "NAF":CLUE.NaiveAdviceFollower(env,trials=trials), # Naive advice follower, with default agent
    "CLUE":CLUE.ClueAgent(env,trials=trials) # CLUE, with default agent
}

'''
PANEL OF EXPERTS
'''
# Dictionary of panels to be tested
panel_dict = {
"Single_Bad":[0],
"Single_Good":[1],
"Varied_Panel":[0,0.1,0.25,0.5,0.75,0.9,1]
}

# Create list of panels
panels = CLUE.make_panels(panel_dict,env)

'''
RUN EXPERIMENT
'''
# Run the panel comparison experiment
rewards,rhos = CLUE.Experiment.panel_comparison(env,agents,panels,trials,runs,display=True)
# TODO: save to csvs
