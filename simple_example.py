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
runs = 10 # Number of runs, each run the agent learns from scratch

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
print("======Running experiment======")
rewards,rhos = CLUE.Experiment.panel_comparison(env,agents,panels,trials,runs,display=True)
# Save results to csv
print("======Saving results======")
CLUE.Experiment.save_panel_comparison_to_csv(rewards,rhos,env,agents,panels,trials,runs)
# Plot results
print("======Plotting graphs======")
base_path = env.name+"/panel_comparison/"+str(trials)+"_trials_"+str(runs)+"_runs/"
panel_titles = {
"Single_Good":"Single Reliable Expert\n($\\rho_{true}=1$)",
"Single_Bad":"Single Unreliable Expert\n($\\rho_{true}=0$)",
"Varied_Panel":"Varied Panel\n($P_{true}=\\{0,0.1,0.25,0.5,0.75,0.9,1\\}$)"
}
reward_range = [-1,1]
CLUE.Plot.plot_reward_comparison(base_path,trials,panel_titles=panel_titles)
