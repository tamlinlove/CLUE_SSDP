import CLUE
import numpy as np

'''
This script runs a simple panel comparison experiment comparing the True Policy,
Baseline, NAF and CLUE agents on a single randomly generated SSDP with 7 state
variables and 3 action variables. Panels compared are the single reliable expert,
single unreliable expert, and varied panel (as in the paper). Results are
averaged over 10 runs, each with 10000 trials, and plotted
'''
experiment_name = "simple"
'''
ENVIRONMENT
'''
# Create a random environment with 7 state variables (|S|=128)
# and 3 action variables (|A|=8)
reward_range = [-1,1]
env = CLUE.make("RandomSSDP",num_chance=7,num_decision=3,reward_range=reward_range)


'''
EXPERIMENT DETAILS
'''
trials = 10000 # Number of trials each run
runs = 10 # Number of runs, each run the agent learns from scratch

'''
AGENTS
'''
# Create some agents, store in dictionary
agent_list = ["True Policy Agent","Baseline Agent","NAF","CLUE","CLUE Decayed"]
agents = CLUE.Experiment.make_agents(agent_list,env,trials)
agents["CLUE"].threshold = 0.25

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
panels = CLUE.Experiment.make_panels(panel_dict,env)

'''
RUN EXPERIMENT
'''
# Run the panel comparison experiment
print("======Running experiment======")
rewards,rhos = CLUE.Experiment.panel_comparison(env,agents,panels,trials,runs,display=True)
# Save results to csv
print("======Saving results======")
CLUE.Experiment.save_panel_comparison_to_csv(rewards,rhos,env.name,agents,panels,trials,runs,directory=experiment_name)
'''
PLOT RESULTS
'''
print("======Plotting graphs======")
# Set path in figures/ directory
base_path = env.name+"/"+experiment_name+"/"+str(trials)+"_trials_"+str(runs)+"_runs/"
# Set titles above each plot
panel_titles = {
"Single_Good":"Single Reliable Expert\n($\\rho_{true}=1$)",
"Single_Bad":"Single Unreliable Expert\n($\\rho_{true}=0$)",
"Varied_Panel":"Varied Panel\n($P_{true}=\\{0,0.1,0.25,0.5,0.75,0.9,1\\}$)"
}
# Plot graphs
CLUE.Plot.plot_reward_comparison(base_path,trials,panel_titles=panel_titles,reward_range=reward_range)
# Plot Rhos
CLUE.Plot.plot_rhos(base_path,trials,panel_titles=panel_titles)
